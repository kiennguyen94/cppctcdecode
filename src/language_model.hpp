#pragma once
#include "constants.hpp"
#include "lm/model.hh"
#include "tsl/htrie_set.h"
#include <lm/state.hh>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace pyctcdecode {
class KenlmState;
using kenlm_state = lm::ngram::State;
using KenlmModel = std::shared_ptr<const lm::ngram::Model>;
using Unigrams = std::unordered_set<std::string>;
using ParamValue = std::variant<float, bool>;

class AbstractLMState {
public:
  virtual ~AbstractLMState() = default;
};

using AbstractLMStatePtr = std::shared_ptr<AbstractLMState>;
using ScoreResult = std::pair<float, AbstractLMStatePtr>;

class KenlmState : public AbstractLMState {
public:
  KenlmState(std::shared_ptr<kenlm_state> state) : state_(std::move(state)){};
  KenlmState(const KenlmState &other) = default;
  KenlmState(KenlmState &&other) = default;
  KenlmState &operator=(const KenlmState &other) = default;
  KenlmState &operator=(KenlmState &&other) = default;
  ~KenlmState() = default;
  std::shared_ptr<kenlm_state> &state() { return state_; }

private:
  std::shared_ptr<kenlm_state> state_;
};

class HotWordScorer;
using HotWordScorerPtr = std::shared_ptr<const HotWordScorer>;
class HotWordScorer {
private:
  std::regex match_ptn_;
  tsl::htrie_set<char> char_trie_;
  float weight_;

public:
  HotWordScorer(std::regex match_ptn, tsl::htrie_set<char> char_trie,
                float weight = DEFAULT_HOTWORD_WEIGHT);
  float score(const std::string &text) const;
  float score_partial_token(const std::string &text) const;
  bool contains(const std::string &item) const;
  static HotWordScorerPtr
  build_scorer(const std::unordered_set<std::string> &hotwords,
               float weight = DEFAULT_HOTWORD_WEIGHT);
};

class AbstractLanguageModel {
public:
  virtual int order() const = 0;
  virtual AbstractLMStatePtr get_start_state() const = 0;
  virtual float score_partial_token(const std::string &partial_token) const = 0;
  virtual ScoreResult score(AbstractLMStatePtr &prev_state,
                            const std::string &word,
                            bool is_last_word = false) const = 0;
};

using AbstractLanguageModelPtr = std::shared_ptr<AbstractLanguageModel>;

class LanguageModel : public AbstractLanguageModel {
public:
  // TODO LanguageModelConfig;
  LanguageModel(KenlmModel kenlm_model,
                std::optional<Unigrams> unigrams = std::nullopt,
                float alpha = DEFAULT_ALPHA, float beta = DEFAULT_BETA,
                float unk_score_offset = DEFAULT_UNK_LOGP_OFFSET,
                bool score_boundary = DEFAULT_SCORE_LM_BOUNDARY);
  void reset_params(const std::unordered_map<std::string, ParamValue> &);
  int order() const override;
  virtual AbstractLMStatePtr get_start_state() const override;
  float score_partial_token(const std::string &) const override;
  ScoreResult score(AbstractLMStatePtr &prev_state, const std::string &word,
                    bool is_last_word = false) const override;

private:
  float get_raw_end_score(std::shared_ptr<kenlm_state> &start_state) const;

private:
  KenlmModel kenlm_model_;
  std::optional<Unigrams> unigram_set_;
  float alpha_;
  float beta_;
  float unk_score_offset_;
  float score_boundary_;
  tsl::htrie_set<char> char_trie_;
};

} // namespace pyctcdecode