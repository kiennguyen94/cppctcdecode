#pragma once
#include "Eigen/Eigen"
#include "alphabet.hpp"
#include "constants.hpp"
#include "language_model.hpp"
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pyctcdecode {

using Frames = std::pair<int, int>;
using WordFrames = std::pair<std::string, Frames>;
using LMScoreCacheKey = std::pair<std::string, bool>;
using LMScoreCacheValue = std::tuple<float, float, AbstractLMStatePtr>;
using LMScoreCache = std::unordered_map<LMScoreCacheKey, LMScoreCacheValue>;

using EigenMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using EigenArray =
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template <int /*axis*/>
void EMatrixLogSoftmax(const EigenMatrix &input, EigenMatrix &output);

struct LMBeam;

struct Beam {
  std::string text_;
  std::string next_word_;
  std::string partial_word_;
  std::optional<std::string> last_char_;
  std::vector<Frames> text_frames_;
  Frames partial_frames_;
  float logit_score_;

  Beam(const Beam &other) = default;
  Beam(Beam &&other) = default;
  Beam &operator=(const Beam &) = default;
  Beam &operator=(Beam &&) = default;
  // Beam();
  static Beam from_lm_beam(const LMBeam &lmbeam);

  // TODO
  void say_hello() const;

  friend std::ostream &operator<<(std::ostream &os, const Beam &beam) {
    os << "[";
    os << "text: [" << beam.text_ << "] ";
    os << "next word: [" << beam.next_word_ << "] ";
    os << "partial word: [" << beam.partial_word_ << "] ";
    os << "last char: ["
       << (beam.last_char_.has_value() ? beam.last_char_.value() : "") << "] ";
    os << "partial frame: [" << beam.partial_frames_.first << ", "
       << beam.partial_frames_.second << "] ";
    os << "logit score: [" << beam.logit_score_ << "]]\n";
    return os;
  }
};

struct LMBeam : Beam {
  float lm_score_;

  bool operator<(const LMBeam &other) const {
    return lm_score_ < other.lm_score_;
  }
  bool operator==(const LMBeam &other) const {
    return lm_score_ == other.lm_score_;
  }
  bool operator>(const LMBeam &other) const {
    return !(*this < other && *this == other);
  }
  bool operator!=(const LMBeam &other) const { return !(*this == other); }

  friend std::ostream &operator<<(std::ostream &os, const LMBeam &beam) {
    os << "[";
    os << "text: [" << beam.text_ << "] ";
    os << "next word: [" << beam.next_word_ << "] ";
    os << "partial word: [" << beam.partial_word_ << "] ";
    os << "last char: ["
       << (beam.last_char_.has_value() ? beam.last_char_.value() : "") << "] ";
    os << "partial frame: [" << beam.partial_frames_.first << ", "
       << beam.partial_frames_.second << "] ";
    os << "logit score: [" << beam.logit_score_ << "] ";
    os << "lm score: [" << beam.lm_score_ << "]]\n";
    return os;
  }
};

struct OutputBeam {
  std::string text_;
  std::optional<AbstractLMStatePtr> last_lm_state;
  std::vector<WordFrames> text_frames;
  float logit_score;
  float lm_score;
};

class BeamSearchDecoderCTC {
private:
  std::vector<LMBeam> get_lm_beam(
      const std::vector<Beam> &beams, const HotWordScorerPtr hotword_scorer,
      LMScoreCache &cached_lm_scores,
      std::unordered_map<std::string, float> &cached_partial_token_scores,
      bool is_eos = false) const;
  std::vector<Beam> partial_decode_logits(
      const Eigen::MatrixXf &logits, std::vector<Beam> &beams, int beam_width,
      float beam_prune_logp, float token_min_logp, bool prune_history,
      const HotWordScorerPtr hotword_scorer, LMScoreCache &cached_lm_scores,
      std::unordered_map<std::string, float> &cached_p_lm_scores,
      int processed_frames = 0) const;

  std::vector<LMBeam>
  finalize_beams(const std::vector<Beam> &beams, int beam_width,
                 float beam_prune_logp, HotWordScorerPtr hotword_scorer,
                 LMScoreCache &cached_lm_scores,
                 std::unordered_map<std::string, float> &cached_p_lm_scores,
                 bool force_next_word = false, bool is_end = false);
  std::vector<OutputBeam> decode_logits(
      const Eigen::MatrixXf &logits, int beam_width, float beam_prune_logp,
      float token_min_logp, bool prune_history, HotWordScorerPtr hotword_scorer,
      std::optional<AbstractLMStatePtr> lm_start_state = std::nullopt);

  void check_logits_dimension(const Eigen::MatrixXf &logits) {
    if (logits.cols() != idx2vocab_.size()) {
      std::stringstream ss;
      ss << "Input logits cols does not match vocab size " << logits.cols()
         << " vs " << idx2vocab_.size();
      throw std::runtime_error(ss.str());
    }
  }

private:
  std::unordered_map<int, AbstractLanguageModelPtr> model_container_;
  AlphabetPtr alphabet_;
  std::unordered_map<size_t, std::string> idx2vocab_;
  bool is_bpe_;
  int model_key_;

public:
  BeamSearchDecoderCTC(
      AlphabetPtr alphabet,
      std::optional<AbstractLanguageModelPtr> language_model = std::nullopt);
  std::vector<Beam> partial_decode_beams(
      const Eigen::MatrixXf &logits, LMScoreCache &cached_lm_scores,
      std::unordered_map<std::string, float> cached_p_lm_scores,
      const std::vector<Beam> &beams, int processed_frames,
      int beam_width = DEFAULT_BEAM_WIDTH,
      float beam_prune_logp = DEFAULT_PRUNE_LOGP,
      float token_min_logp = DEFAULT_MIN_TOKEN_LOGP,
      bool prune_history = DEFAULT_PRUNE_BEAMS,
      HotWordScorerPtr hotword_scorer = nullptr, bool force_next_word = false,
      bool is_end = false);

  std::vector<OutputBeam>
  decode_beams(Eigen::MatrixXf &logits, int beam_width = DEFAULT_BEAM_WIDTH,
               float beam_prune_logp = DEFAULT_PRUNE_LOGP,
               float token_min_logp = DEFAULT_MIN_TOKEN_LOGP,
               bool prune_history = DEFAULT_PRUNE_BEAMS,
               const std::unordered_set<std::string> &hotwords = {},
               float hotword_weight = DEFAULT_HOTWORD_WEIGHT,
               std::optional<AbstractLMState> lm_start_state = std::nullopt);

  std::string
  decode(const Eigen::MatrixXf &logits, int beam_width = DEFAULT_BEAM_WIDTH,
         float beam_prune_logp = DEFAULT_PRUNE_LOGP,
         float token_min_logp = DEFAULT_MIN_TOKEN_LOGP,
         bool prune_history = DEFAULT_PRUNE_BEAMS,
         const std::unordered_set<std::string> &hotwords = {},
         float hotword_weight = DEFAULT_HOTWORD_WEIGHT,
         std::optional<AbstractLMState> lm_start_state = std::nullopt);
};

using BeamSearchDecoderCTCPtr = std::shared_ptr<BeamSearchDecoderCTC>;

BeamSearchDecoderCTCPtr build_ctcdecoder(
    const Labels &labels,
    std::optional<std::filesystem::path> kenlm_model_path = std::nullopt,
    std::optional<Unigrams> unigrams = std::nullopt,
    float alpha = DEFAULT_ALPHA, float beta = DEFAULT_BETA,
    float unk_score_offset = DEFAULT_UNK_LOGP_OFFSET,
    bool lm_score_boundary = DEFAULT_SCORE_LM_BOUNDARY);
} // namespace pyctcdecode