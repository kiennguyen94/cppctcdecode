#include "alphabet.hpp"
#include "constants.hpp"
#include <algorithm>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

bool check_if_bpe(const pyctcdecode::Labels &labels) {
  const auto is_bpe =
      std::any_of(labels.cbegin(), labels.cend(), [](const std::string &item) {
        return (item.find(pyctcdecode::BPE_TOKEN) == 0u) ||
               (item.find(pyctcdecode::BPE_TOKEN_ALT) == 0u);
      });
  if (is_bpe) {
    printf("Alphabet is bpe\n");
  } else {
    printf("Alphabet is regular, not bpe\n");
  }
  return is_bpe;
}

pyctcdecode::Labels
normalize_regular_alphabet(const pyctcdecode::Labels &labels) {
  auto normalized_labels = labels;
  // substitute space
  auto pipe_pos =
      std::find(normalized_labels.begin(), normalized_labels.end(), "|");
  if (pipe_pos != normalized_labels.end() &&
      std::find(normalized_labels.cbegin(), normalized_labels.cend(), " ") ==
          normalized_labels.end()) {
    printf("Found | in vocab but not ' ', subtituting\n");
    *pipe_pos = " ";
  }
  // substitute ctc blank char
  std::smatch base_match;
  for (auto idx = 0; idx < normalized_labels.size(); idx++) {
    if (std::regex_search(normalized_labels.at(idx), base_match,
                          pyctcdecode::BLANK_TOKEN_PTN)) {
      printf("Found [%s] in vocab, replacing with empty string\n",
             normalized_labels.at(idx).c_str());
      normalized_labels.at(idx) = "";
    }
  }
  auto pos = std::find(normalized_labels.begin(), normalized_labels.end(), "_");
  if (pos != normalized_labels.end() &&
      std::find(normalized_labels.cbegin(), normalized_labels.cend(), "") ==
          normalized_labels.end()) {
    *pos = "";
  }
  if (std::find(normalized_labels.cbegin(), normalized_labels.cend(), "") ==
      normalized_labels.end()) {
    printf("appending blank char\n");
    normalized_labels.push_back("");
  }
  // substitute unk
  for (auto idx = 0; idx < normalized_labels.size(); idx++) {
    if (std::regex_search(normalized_labels.at(idx), base_match,
                          pyctcdecode::UNK_TOKEN_PTN)) {
      printf("Found [%s] in vocab, replacing with [%s]\n",
             normalized_labels.at(idx).c_str(), pyctcdecode::UNK_TOKEN.c_str());
      normalized_labels[idx] = pyctcdecode::UNK_TOKEN;
    }
  }
  // additional checks
  if (std::any_of(normalized_labels.cbegin(), normalized_labels.cend(),
                  [](const auto &item) { return item.size() > 1; })) {
    printf("Found entries of length > 1 in alphabet. This is unusual "
           "unless style is BPE, but the "
           "alphabet was not recognized as BPE type. Is this correct?");
  }

  pos = std::find(normalized_labels.begin(), normalized_labels.end(), " ");
  if (pos == normalized_labels.end()) {
    printf("Space token ' ' missing from vocabulary.\n");
  }
  // DEBUG
  std::stringstream ss;
  for (const auto &n : normalized_labels) {
    ss << n << ",";
  }
  printf("normalized labels [%s]\n", ss.str().c_str());
  return normalized_labels;
}
} // namespace

namespace pyctcdecode {
Alphabet::Alphabet(Labels labels, bool is_bpe)
    : labels_(std::move(labels)), is_bpe_(is_bpe) {}

AlphabetPtr Alphabet::build_alphabet(const Labels &labels) {
  const auto is_bpe = check_if_bpe(labels);
  // TODO verify alphabet
  if (is_bpe) {
    throw std::runtime_error("Normalizing bpe alphabet not implemented\n");
  } else {
    return std::make_shared<Alphabet>(
        Alphabet(normalize_regular_alphabet(labels), is_bpe));
  }
}
} // namespace pyctcdecode