# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import Levenshtein as Lev


def __levenshtein(a: List, b: List) -> int:
    """Calculates the Levenshtein distance between a and b.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

'''
def word_error_rate(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same length.

    Args:
        hypotheses: list of hypotheses
        references: list of references

    Returns:
        (float) average word error rate
    """
    scores = 0
    words = 0
    len_diff = len(references) - len(hypotheses) 
    if len_diff > 0:
        raise ValueError("In word error rate calculation, hypotheses and reference"
                         " lists must have the same number of elements. But I got:"
                         "{0} and {1} correspondingly".format(len(hypotheses), len(references)))
    elif len_diff < 0:
        hypotheses = hypotheses[:len_diff]
    
    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
    if words!=0:
        wer = 1.0*scores/words
    else:
        wer = float('inf')
    return wer, scores, words
'''


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    wer_list = []
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
        wer_list.append((1.0*__levenshtein(h_list, r_list))/(len(r_list)+1e-20))
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer, wer_list, scores, words

def f_wer(transcript, reference):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        transcript (string): space-separated sentence
        reference (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(transcript.split() + reference.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in transcript.split()]
    w2 = [chr(word2char[w]) for w in reference.split()]

    return float(Lev.distance(''.join(w1), ''.join(w2)))/(len(reference.split())+1e-20)

def f_cer(transcript, reference):
    """
    Computes the Character Error Rate, defined as the edit distan
    Arguments:
        transcript (string): space-separated sentence
        reference (string): space-separated sentence
    """
    transcript, reference = transcript.replace(' ', ''), reference.replace(' ', '')
    return float(Lev.distance(transcript, reference))/len(reference)