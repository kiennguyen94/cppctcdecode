#include "language_model.hpp"
#include "constants.hpp"
#include "tsl/htrie_set.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <limits>
#include <lm/state.hh>
#include <memory>
#include <regex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
// pyctcdecode::Unigrams prepare_unigram_set()
}

namespace pyctcdecode {

HotWordScorer::HotWordScorer(std::regex match_ptn,
                             tsl::htrie_set<char> char_trie, float weight)
    : match_ptn_(std::move(match_ptn)), char_trie_(std::move(char_trie)),
      weight_(weight) {}

float HotWordScorer::score(const std::string &text) const {
  std::smatch matches;
  std::regex_search(text, matches, match_ptn_);
  return matches.size() * weight_;
}

float HotWordScorer::score_partial_token(const std::string &text) const {
  if (this->contains(text)) {
    auto min_len = std::numeric_limits<size_t>::max();
    auto prefix_range = char_trie_.equal_prefix_range(text);
    std::string key_buffer;
    for (auto &it = prefix_range.first; it != prefix_range.second; it++) {
      it.key(key_buffer);
      min_len = std::min(min_len, key_buffer.size());
    }
    return weight_ * ((float)text.size() / min_len);
  }
  return 0.0f;
}

bool HotWordScorer::contains(const std::string &item) const {
  const auto prefix_range = char_trie_.equal_prefix_range(item);
  return prefix_range.first == prefix_range.second ? false : true;
}

HotWordScorerPtr
HotWordScorer::build_scorer(const std::unordered_set<std::string> &hotwords,
                            float weight) {
  std::vector<std::string> hotwords_strip;
  std::transform(
      hotwords.begin(), hotwords.end(), std::back_inserter(hotwords_strip),
      [](const auto &item) { return boost::algorithm::trim_copy(item); });
  if (!hotwords_strip.empty()) {
    std::vector<std::string> hotword_unigrams;
    for (const auto &ngram : hotwords_strip) {
      std::vector<std::string> results;
      boost::split(results, ngram, boost::is_any_of(" "));
      for (const auto &unigram : results) {
        hotword_unigrams.emplace_back(unigram);
      }
    }
    std::sort(hotword_unigrams.begin(), hotword_unigrams.end());
    tsl::htrie_set<char> char_trie;
    for (const auto &item : boost::adaptors::reverse(hotword_unigrams)) {
      char_trie.insert(item);
    }
    std::transform(hotword_unigrams.begin(), hotword_unigrams.end(),
                   hotword_unigrams.begin(), [](std::string &item) {
                     return R"((?<!\S))" + item + R"((?!\S))";
                   });
    std::regex match_ptn(boost::join(hotword_unigrams, "|"));
    return std::make_shared<HotWordScorer>(match_ptn, char_trie);
  }
  return std::make_shared<HotWordScorer>(std::regex(R"(^\b$)"),
                                         tsl::htrie_set<char>());
}

LanguageModel::LanguageModel(KenlmModel kenlm_model,
                             std::optional<Unigrams> unigrams, float alpha,
                             float beta, float unk_score_offset,
                             bool score_boundary)
    : kenlm_model_(kenlm_model), unigram_set_(unigrams), alpha_(alpha),
      beta_(beta), unk_score_offset_(unk_score_offset),
      score_boundary_(score_boundary) {
  // TODO distinguish between optional nullopt and empty set
  if (unigram_set_->empty()) {
    // printf("No unigrams\n");
  } else {
    // build char trie from unigram set
    // TODO reserving?
    for (const auto &word : unigram_set_.value()) {
      char_trie_.insert(word.c_str());
    }
  }
}

int LanguageModel::order() const { return (int)(kenlm_model_->Order()); }

AbstractLMStatePtr LanguageModel::get_start_state() const {
  auto start_state = std::make_shared<kenlm_state>();
  if (score_boundary_) {
    // printf("get start state score boundary\n");
    kenlm_model_->BeginSentenceWrite(start_state.get());
  } else {
    // printf("get start state null context write\n");
    kenlm_model_->NullContextWrite(start_state.get());
  }
  return std::make_shared<KenlmState>(start_state);
}

float LanguageModel::get_raw_end_score(
    std::shared_ptr<kenlm_state> &start_state) const {
  if (score_boundary_) {
    auto end_state = std::make_shared<kenlm_state>();
    return kenlm_model_->BaseScore(start_state.get(),
                                   kenlm_model_->BaseVocabulary().Index("</s>"),
                                   end_state.get());
  } else {
    return 0.0;
  }
}

float LanguageModel::score_partial_token(
    const std::string &partial_token) const {
  float is_oov;
  if (char_trie_.empty()) {
    is_oov = 1.0;
  } else {
    auto range = char_trie_.equal_prefix_range(partial_token);
    is_oov = range.first == range.second ? 1 : 0;
  }
  float unk_score = unk_score_offset_ * is_oov;
  if (partial_token.size() > AVG_TOKEN_LEN) {
    unk_score = unk_score * (float)partial_token.size() / AVG_TOKEN_LEN;
  }
  return unk_score;
}

ScoreResult LanguageModel::score(AbstractLMStatePtr &prev_state_,
                                 const std::string &word,
                                 bool is_last_word) const {
  auto end_state = std::make_shared<kenlm_state>();
  // static case checks at compile time
  // auto prev_state = static_cast<KenlmState *>(&prev_state_);
  auto prev_state = std::dynamic_pointer_cast<KenlmState>(prev_state_);
  auto lm_score = kenlm_model_->BaseScore(
      (prev_state->state().get()), kenlm_model_->BaseVocabulary().Index(word),
      end_state.get());
  // printf("xxx word [%s] index [%d] lm score [%f]\n", word.c_str(),
  //  kenlm_model_->BaseVocabulary().Index(word), lm_score);
  auto start = lm::ngram::State();
  auto end = lm::ngram::State();
  kenlm_model_->BeginSentenceWrite(&start);
  auto other = start;
  // printf("alternate score [%f]\n",
  //  kenlm_model_->BaseScore(
  //      &other, kenlm_model_->BaseVocabulary().Index(word), &end));
  if (unigram_set_->size() > 0 && unigram_set_.value().count(word) == 0 ||
      kenlm_model_->BaseVocabulary().Index(word) == 0) {
    lm_score += unk_score_offset_;
  }
  if (is_last_word) {
    lm_score += get_raw_end_score(end_state);
  }
  lm_score = alpha_ * lm_score * LOG_BASE_CHANGE_FACTOR + beta_;
  return std::make_pair(lm_score, std::make_shared<KenlmState>(end_state));
}
} // namespace pyctcdecode