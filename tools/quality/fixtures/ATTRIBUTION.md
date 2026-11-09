# Corpus attribution

Third-party corpus text retains the following source licenses. Inclusion here
does not relicense it under Xenolith's code license. Exact dataset revisions,
record identifiers, text checksums and token sequences are in the corpus
manifest. Text is selected from the pinned datasets; no text is repeated to
create a longer input. Tokenization adds BOS 2, without a chat template.

| Fixture | Source and authors | License and transformations |
| --- | --- | --- |
| `wiki-du-fu` | Wikipedia contributors, [Du Fu and revision history](https://en.wikipedia.org/w/index.php?title=Du_Fu&action=history), distributed in WikiText by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); the WikiText-normalized article is retained, including its spacing and headings. |
| `wiki-frank-headlam` | Wikipedia contributors, [Frank Headlam and revision history](https://en.wikipedia.org/w/index.php?title=Frank_Headlam&action=history), distributed in WikiText by the same authors | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); normalized article retained, numerical input limited to its first 2,048 tokens including BOS. |
| `instruction-json` | Google Research, IFEval authors; prompt key 13 on Prospect Park and JSON formatting | [Apache 2.0, included](licenses/Google-Research-LICENSE.txt); prompt text unchanged. |
| `python-close-elements` | OpenAI HumanEval contributors and EvalPlus contributors; `HumanEval/0`, `has_close_elements` | [HumanEval MIT, included](licenses/HumanEval-LICENSE.txt), and [HumanEval+ Apache 2.0 with its MIT notice, included](licenses/HumanEvalPlus-LICENSE.txt); prompt unchanged. No solutions or generated tests are used. |
| `italian-accordion` | Lucas Bandarkar and the Belebele authors; Wikibooks contributors, [Accordion/Right hand and history](https://en.wikibooks.org/w/index.php?title=Accordion/Right_hand&action=history); Italian passage/question 1 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); source passage, question and answer choices assembled as a plain-text input. |

Pinned source distributions:

- [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md), revision `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1`; Du Fu from test, Frank Headlam from validation. The card also lists GFDL; this distribution uses the CC BY-SA option.
- [IFEval data](https://github.com/google-research/google-research/blob/26d8ccdab6fec61b5c83ad6327ea8bda9e580288/instruction_following_eval/data/input_data.jsonl), revision `26d8ccdab6fec61b5c83ad6327ea8bda9e580288`.
- [HumanEval+ release](https://github.com/evalplus/humanevalplus_release/tree/68cd26d53a0dec69f85eafe1f82a2a74155a2bd6), revision `68cd26d53a0dec69f85eafe1f82a2a74155a2bd6`, release 0.1.10.
- [Belebele dataset card](https://huggingface.co/datasets/facebook/belebele/blob/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/README.md), revision `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`, `ita_Latn`, test. See Bandarkar et al., [The Belebele Benchmark](https://aclanthology.org/2024.acl-long.44/), ACL 2024.

These are numerical test inputs. The suite does not report the original
benchmarks' instruction-following, code-generation or comprehension scores.
