"""
SeMaScore (Semantic Match Score) generator.

Based on: https://github.com/zenlab-edgeASR/SeMaScore/tree/main/codes
"""

from collections import defaultdict
from packaging import version

import torch
import re
from transformers import AutoModel, AutoTokenizer, GPT2Tokenizer, RobertaTokenizer
from transformers import __version__ as trans_version
from transformers import logging

logging.set_verbosity_error()


def get_model(model_type, num_layers, all_layers=None):
    """Load and configure a pretrained transformer model."""
    model = AutoModel.from_pretrained(model_type)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    # Drop unused layers
    if not all_layers:
        if hasattr(model, "n_layers"):
            assert (
                0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):
            assert (
                0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList(
                [layer for layer in model.layer[:num_layers]]
            )
        elif hasattr(model, "encoder"):
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):
                assert (
                    0 <= num_layers <= len(model.encoder.block)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.block)} for {model_type}"
                model.encoder.block = torch.nn.ModuleList(
                    [layer for layer in model.encoder.block[:num_layers]]
                )
            else:
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList(
                    [layer for layer in model.encoder.layer[:num_layers]]
                )
        elif hasattr(model, "transformer"):
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList(
                [layer for layer in model.transformer.layer[:num_layers]]
            )
        elif hasattr(model, "layers"):
            assert (
                0 <= num_layers <= len(model.layers)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layers)} for {model_type}"
            model.layers = torch.nn.ModuleList(
                [layer for layer in model.layers[:num_layers]]
            )
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True

    return model


def get_tokenizer(model_type, use_fast=False):
    """Load a pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=use_fast)
    return tokenizer


model_type = "roberta-base"
num_layers = 12
tokenizer = get_tokenizer(model_type, use_fast=False)
model = get_model(model_type, num_layers)


def padding(arr, pad_token, dtype=torch.long):
    """Pad a list of token arrays to uniform length."""
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask


def collate_idf(arr, tokenizer, idf_dict, device="cuda:0"):
    """
    Pad a list of sentences to the same length and load IDF scores for their tokens.

    Args:
        arr (list of str): Sentences to process.
        tokenizer: Tokenizer to encode sentences.
        idf_dict (dict): Mapping from word piece index to its inverse document frequency.
        device (str): Device to use, e.g. 'cpu' or 'cuda'.
    """
    arr = [sent_encode(tokenizer, a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask


def bert_encode(model, x, attention_mask, all_layers=False):
    """Encode input tokens through a BERT-like model."""
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask, output_hidden_states=all_layers)
    if all_layers:
        emb = torch.stack(out[-1], dim=2)
    else:
        emb = out[0]
    return emb


def get_bert_embedding(
    all_sens,
    model,
    tokenizer,
    idf_dict,
    batch_size=-1,
    device="cuda:0",
    all_layers=False,
):
    """
    Compute BERT embedding in batches.

    Args:
        all_sens (list of str): Sentences to encode.
        model: A BERT-like model.
        tokenizer: A tokenizer corresponding to the model.
        idf_dict (dict): Mapping from word piece index to its inverse document frequency.
        device (str): Device to use, e.g. 'cpu' or 'cuda'.
    """
    padded_sens, padded_idf, lens, mask = collate_idf(
        all_sens, tokenizer, idf_dict, device=device
    )

    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(
                model,
                padded_sens[i : i + batch_size],
                attention_mask=mask[i : i + batch_size],
                all_layers=all_layers,
            )
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)

    return total_embedding, mask, padded_idf


def sent_encode(tokenizer, sent):
    """Encode a sentence using the given tokenizer."""
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    elif isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, RobertaTokenizer):
        if version.parse(trans_version) >= version.parse("4.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=512,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse("3.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=512,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse("2.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=512,
            )
        else:
            raise NotImplementedError(
                f"transformers version {trans_version} is not supported"
            )
    else:
        if version.parse(trans_version) >= version.parse("4.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse("3.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse("2.0.0"):
            return tokenizer.encode(
                sent, add_special_tokens=True, max_length=512
            )
        else:
            raise NotImplementedError(
                f"transformers version {trans_version} is not supported"
            )


idf_dict = defaultdict(lambda: 1.0)
idf_dict[tokenizer.sep_token_id] = 0
idf_dict[tokenizer.cls_token_id] = 0

device = 'cpu'


def printChanges(s1, s2, dp):
    """Get the transformed ground truth after applying edit-distance."""
    x = []
    i = len(s1)
    j = len(s2)

    while(i > 0 and j > 0):
        if s1[i - 1] == s2[j - 1]:
            x.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            j -= 1
            i -= 1
            x.append('$')
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
            x.append('-')
        elif dp[i][j] == dp[i][j - 1] + 1:
            j -= 1
            x.append('+')
    while i > 0:
        x.append('-')
        i -= 1
    while j > 0:
        x.append('+')
        j -= 1
    return x


def editDP(s1, s2):
    """Compute edit-distance DP matrix and return aligned operations."""
    len1 = len(s1)
    len2 = len(s2)
    dp = [[0 for i in range(len2 + 1)]
             for j in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s2[j - 1] == s1[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],
                                   dp[i - 1][j - 1],
                                   dp[i - 1][j])

    x = printChanges(s1, s2, dp)
    return x


def hit_values(ref_li, hyp_li, verbose=False):
    """Call the edit-distance function and return the aligned string."""
    x = editDP(hyp_li, ref_li)
    if verbose:
        print(ref_li)
        print(hyp_li)
    aligned = x[::-1]
    aligned = ''.join(map(str, aligned))
    if verbose:
        print(aligned)
    return aligned


def sub_sentence_mapper1(ground_truth, aligned, verbose=False):
    """Map ground truth segments with the aligned representation (deletion-based)."""
    split_ground_truth_v1 = []
    split_aligned_v1 = []
    count = 0
    index = 0
    temp_str1, temp_str2 = '', ''
    while count < len(ground_truth) and index < len(aligned):
        if ground_truth[count] == ' ' and aligned[index] == ' ':
            split_ground_truth_v1.append(temp_str1)
            split_aligned_v1.append(temp_str2)
            temp_str1 = ''
            temp_str2 = ''
            count += 1
            index += 1
        else:
            if aligned[index] == '-':
                temp_str2 += aligned[index]
                index += 1
            else:
                temp_str1 += ground_truth[count]
                temp_str2 += aligned[index]
                count += 1
                index += 1

    while count < len(ground_truth):
        temp_str1 += ground_truth[count]
        count += 1

    while index < len(aligned):
        temp_str2 += aligned[index]
        index += 1
    split_ground_truth_v1.append(temp_str1)
    split_aligned_v1.append(temp_str2)
    if verbose:
        print(f'#{split_ground_truth_v1}')
        print(f'#{split_aligned_v1}')
    return split_ground_truth_v1, len(split_ground_truth_v1) == len(split_aligned_v1)


def sub_sentence_mapper2(ground_truth, aligned, verbose=False):
    """Map hypothesis segments with the aligned representation (insertion-based)."""
    split_ground_truth_v1 = []
    split_aligned_v1 = []
    count = 0
    index = 0
    temp_str1, temp_str2 = '', ''
    while count < len(ground_truth) and index < len(aligned):
        if ground_truth[count] == ' ' and aligned[index] == ' ':
            split_ground_truth_v1.append(temp_str1)
            split_aligned_v1.append(temp_str2)
            temp_str1 = ''
            temp_str2 = ''
            count += 1
            index += 1
        else:
            if aligned[index] == '+':
                temp_str2 += aligned[index]
                index += 1
            else:
                temp_str1 += ground_truth[count]
                temp_str2 += aligned[index]
                count += 1
                index += 1

    while count < len(ground_truth):
        temp_str1 += ground_truth[count]
        count += 1

    while index < len(aligned):
        temp_str2 += aligned[index]
        index += 1
    split_ground_truth_v1.append(temp_str1)
    split_aligned_v1.append(temp_str2)
    if verbose:
        print(f'#{split_ground_truth_v1}')
        print(f'#{split_aligned_v1}')
        print(len(split_ground_truth_v1) == len(split_aligned_v1))
    return split_ground_truth_v1, len(split_ground_truth_v1) == len(split_aligned_v1)


def get_mer(ground_truth, inference, verbose=False):
    """Compute (1 - match error rate)."""
    aligned = hit_values(ground_truth, inference, verbose=verbose)
    mismatches = 0
    for i in aligned:
        if i in ('+', '-', '$'):
            mismatches += 1
    mer = mismatches / max(len(ground_truth), len(inference))
    return 1 - mer


def mapped_sentence(ground_truth, inference, verbose=False):
    """Get the aligned ground truth and hypothesis along with (1 - match error rate)."""
    aligned = hit_values(ground_truth, inference, verbose=verbose)
    mismatches = 0
    for i in aligned:
        if i in ('+', '-', '$'):
            mismatches += 1
    mer = mismatches / max(len(ground_truth), len(inference))

    mapped_ground_truth, mapped_ground_truth_res = sub_sentence_mapper1(ground_truth, aligned, verbose=verbose)
    mapped_inference, mapped_inference_res = sub_sentence_mapper2(inference, aligned, verbose=verbose)
    if mapped_ground_truth_res and mapped_inference_res:
        return mapped_ground_truth, mapped_inference, aligned, mer
    else:
        return False, False, False, False


def cos_sim(a, b, verbose=False):
    """Compute cosine similarity between two tensors."""
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    ss = torch.mm(a_norm, b_norm.transpose(0, 1)).item()
    if verbose:
        print(f'SS:{ss}')
    return ss


def get_gt_embeddings_v1(ground_truth_v1, gt_embedding, gt_tokens, verbose=False):
    """Get embeddings for each segment of the aligned ground truth."""
    start, end, k = 0, 0, 0
    gt_embeddings_v1 = []
    word = ''
    test_gt = []
    for j in range(len(ground_truth_v1)):
        word = ''
        start = k + 1
        while k < len(gt_tokens):
            word += gt_tokens[k]
            if verbose:
                print(f'word:{word},ground_truth_v1[j]:{ground_truth_v1[j]}')
            if word.strip() == ground_truth_v1[j].strip():
                test_gt.append(word)
                gt_embeddings_v1.append(torch.mean(gt_embedding[0][start:k+2], dim=0))
                if verbose:
                    print((start, k+2))
                k += 1
                break
            k += 1
    if verbose:
        print(test_gt)
        print(len(gt_embeddings_v1), len(ground_truth_v1))
    return gt_embeddings_v1


def get_hyp_embeddings_v1(aligned_v1, hyp_embedding, hyp_tokens, verbose=False):
    """Get embeddings for each segment of the aligned hypothesis."""
    start, end, k = 0, 0, 0
    hyp_embeddings_v1 = []
    word = ''
    test_hyp = []
    for j in range(len(aligned_v1)):
        word = ''
        start = k + 1
        while k < len(hyp_tokens):
            word += hyp_tokens[k]
            if verbose:
                print(f'word:{word},aligned_v1[j]:{aligned_v1[j]}')
            if word.strip() == aligned_v1[j].strip():
                test_hyp.append(word)
                hyp_embeddings_v1.append(torch.mean(hyp_embedding[0][start:k+2], dim=0))
                if verbose:
                    print((start, k+2))
                k += 1
                break
            k += 1
    if verbose:
        print(test_hyp)
        print(len(hyp_embeddings_v1), len(aligned_v1))
    return hyp_embeddings_v1


def generate_sema_score(ground_truth, hypothesis, verbose=False):
    """
    Generate the SeMaScore for a ground truth and hypothesis pair.

    Args:
        ground_truth (str): Reference transcription.
        hypothesis (str): Predicted transcription.
        verbose (bool): Enable detailed debug output.

    Returns:
        tuple: (score, ground_truth_segments, hypothesis_segments, aligned,
                similarity_list, importance_list, multiplication_list, mer_list)
    """
    ground_truth = re.sub(r'[^\w\s]', '', ground_truth.lower())
    hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower())
    gt_embedding, masks, padded_idf = get_bert_embedding(
        [ground_truth], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    hyp_embedding, masks, padded_idf = get_bert_embedding(
        [hypothesis], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ground_truth_v1, aligned_v1, aligned, mer = mapped_sentence(
        ground_truth, hypothesis, verbose=verbose
    )
    gt_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, ground_truth)][1:-1]
    hyp_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, hypothesis)][1:-1]
    if verbose:
        print(ground_truth_v1)
        print(gt_tokens)
        print(aligned_v1)
        print(hyp_tokens)

    gt_embeddings_v1 = get_gt_embeddings_v1(
        ground_truth_v1, gt_embedding, gt_tokens, verbose=verbose
    )
    hyp_embeddings_v1 = get_hyp_embeddings_v1(
        aligned_v1, hyp_embedding, hyp_tokens, verbose=verbose
    )
    total_gt_embedding = torch.mean(gt_embedding[0][1:-1], dim=0)
    ss_list = []
    importance_list = []
    multiplication_list = []
    mer_list = []
    average = 0
    metric = 0
    for j in range(len(gt_embeddings_v1)):
        mer_word = 0
        importance = cos_sim(gt_embeddings_v1[j], total_gt_embedding, verbose=verbose)
        ss = cos_sim(gt_embeddings_v1[j], hyp_embeddings_v1[j], verbose=verbose)
        importance = (importance + 1) / 2
        ss = (ss + 1) / 2
        mer_word = get_mer(ground_truth_v1[j], aligned_v1[j], verbose=verbose)
        mer_list.append(mer_word)
        metric += importance * ss * mer_word
        average += importance
        ss_list.append(round(ss, 4))
        importance_list.append(round(importance, 4))
        multiplication_list.append(round(importance * ss, 4))
        if verbose:
            print(f'{round(ss, 4)}#{ground_truth_v1[j]}#{aligned_v1[j]}')
            print(f'{round(importance, 4)}#{ground_truth_v1[j]}#{ground_truth}')
    metric /= average
    if verbose:
        print(metric)
    return (metric, ground_truth_v1, aligned_v1, aligned, ss_list, importance_list, multiplication_list, mer_list)
