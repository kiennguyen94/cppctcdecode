#include "decoder.hpp"
#include "alphabet.hpp"
#include "constants.hpp"
#include "language_model.hpp"
#include <__algorithm/remove_if.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/functional/hash.hpp>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
struct beam_prefix {
  std::string text;
  std::string partial_word;
  std::optional<std::string> last_char;

  bool operator==(const beam_prefix &other) const {
    return text == other.text && partial_word == other.partial_word &&
           last_char == other.last_char;
  }
};

struct BeamComp {
  bool operator()(const pyctcdecode::Beam &beam1,
                  const pyctcdecode::Beam &beam2) const {
    return static_cast<const pyctcdecode::LMBeam *>(&beam1)->lm_score_ <
           static_cast<const pyctcdecode::LMBeam *>(&beam2)->lm_score_;
  }
};

struct beam_history_prefix {
  std::vector<std::string> text;
  std::string partial_word;
  std::optional<std::string> last_char;
  bool operator==(const beam_history_prefix &other) const {
    return text == other.text && partial_word == other.partial_word &&
           last_char == other.last_char;
  }
};

std::vector<pyctcdecode::LMBeam>
sort_and_trim_beams(std::vector<pyctcdecode::LMBeam> &beams, int beam_width) {
  std::vector<pyctcdecode::LMBeam> rv;
  std::priority_queue beam_pq(beams.begin(), beams.end());
  for (auto i = 0; i < std::min((size_t)beam_width, beams.size()); i++) {
    auto top = beam_pq.top();
    rv.push_back(top);
    beam_pq.pop();
  }
  return rv;
}

std::string normalize_whitespace(const std::string &text) {
  std::vector<std::string> split_text;
  boost::split(split_text, text, boost::is_any_of(" "));
  return boost::algorithm::join(split_text, " ");
}

} // namespace

namespace std {
template <> struct std::hash<pyctcdecode::LMScoreCacheKey> {
  std::size_t operator()(const pyctcdecode::LMScoreCacheKey &key) const {
    return std::hash<string>()(key.first) ^ std::hash<bool>()(key.second);
  }
};

template <> struct std::hash<beam_prefix> {
  size_t operator()(const beam_prefix &key) const {
    return std::hash<string>()(key.text) ^
           std::hash<string>()(key.partial_word) ^
           std::hash<std::optional<string>>()(key.last_char);
  }
};

template <> struct std::hash<beam_history_prefix> {
  size_t operator()(const beam_history_prefix &key) const {
    size_t seed = 0;
    boost::hash_combine(seed,
                        boost::hash_range(key.text.begin(), key.text.end()));
    boost::hash_combine(seed, key.last_char);
    boost::hash_combine(seed, key.partial_word);
    return seed;
  }
};

} // namespace std

namespace {
std::string merge_token(const std::string &token_1,
                        const std::string &token_2) {
  if (token_2.empty()) {
    return token_1;
  } else if (token_1.empty()) {
    return token_2;
  } else {
    return token_1 + " " + token_2;
  }
}

float sum_log_scores(float s1, float s2) {
  if (s1 >= s2) {
    return s1 + log(1 + exp(s2 - s1));
  }
  return s2 + log(1 + exp(s1 - s2));
}

const pyctcdecode::Frames NULL_FRAMES{-1, -1};

std::vector<pyctcdecode::Beam>
merge_beams(const std::vector<pyctcdecode::Beam> &beams) {
  using BeamPrefixDict = std::unordered_map<beam_prefix, pyctcdecode::Beam>;
  std::vector<pyctcdecode::Beam> rv;
  BeamPrefixDict beam_dict;
  for (const auto &beam : beams) {
    const auto new_text = merge_token(beam.text_, beam.next_word_);
    const beam_prefix hash_idx{new_text, beam.partial_word_, beam.last_char_};
    // NB: if does not exist, insert beam
    // if exists, modify score
    // it is pointer to the inserted element or element that prevents inserting
    const auto [it, insterted] =
        beam_dict.insert(std::make_pair(hash_idx, beam));
    if (!insterted) {
      it->second = pyctcdecode::Beam(beam);
      const auto previous_score = it->second.logit_score_;
      it->second.logit_score_ =
          sum_log_scores(previous_score, beam.logit_score_);
    }
  }
  rv.reserve(beam_dict.size());
  std::transform(beam_dict.begin(), beam_dict.end(), std::back_inserter(rv),
                 [](const auto &item) { return item.second; });
  return rv;
}

std::vector<pyctcdecode::Beam>
do_prune_history(const std::vector<pyctcdecode::LMBeam> &beams, int lm_order) {
  const auto min_n_history = std::max(1, lm_order - 1);
  std::unordered_set<beam_history_prefix> seen_hashes;
  std::vector<pyctcdecode::Beam> filtered_beams;
  for (const auto &beam : beams) {
    std::vector<std::string> split_text;
    boost::split(split_text, beam.text_, boost::is_any_of(" "));
    split_text.erase(split_text.end() - min_n_history, split_text.end());
    const beam_history_prefix hash_idx{split_text, beam.partial_word_,
                                       beam.last_char_};
    const auto [it, inserted] = seen_hashes.insert(hash_idx);
    if (inserted) {
      filtered_beams.push_back(pyctcdecode::Beam::from_lm_beam(beam));
    }
  }
  return filtered_beams;
}

const pyctcdecode::Beam EMPTY_START_BEAM{"", "",          "", std::nullopt,
                                         {}, NULL_FRAMES, 0.0};
} // namespace

namespace pyctcdecode {

Beam Beam::from_lm_beam(const LMBeam &lmbeam) {
  return Beam{lmbeam.text_,       lmbeam.next_word_,   lmbeam.partial_word_,
              lmbeam.last_char_,  lmbeam.text_frames_, lmbeam.partial_frames_,
              lmbeam.logit_score_};
}

template <>
void EMatrixLogSoftmax<1>(const EigenMatrix &input, EigenMatrix &output) {
  EigenArray wMinusMax = input.colwise() - input.rowwise().maxCoeff();
  output = wMinusMax.colwise() - wMinusMax.exp().rowwise().sum().log();
}

template <>
void EMatrixLogSoftmax<0>(const EigenMatrix &input, EigenMatrix &output) {
  EigenArray wMinusMax = input.rowwise() - input.colwise().maxCoeff();
  output = wMinusMax.rowwise() - wMinusMax.exp().colwise().sum().log();
}

void Beam::say_hello() const { printf("text [%s]\n", text_.c_str()); }

Beam from_lm_beam(const LMBeam &lmbeam) {
  return Beam{lmbeam.text_,       lmbeam.next_word_,   lmbeam.partial_word_,
              lmbeam.last_char_,  lmbeam.text_frames_, lmbeam.partial_frames_,
              lmbeam.logit_score_};
}

BeamSearchDecoderCTC::BeamSearchDecoderCTC(
    AlphabetPtr alphabet,
    std::optional<AbstractLanguageModelPtr> language_model)
    : alphabet_(std::move(alphabet)), is_bpe_(alphabet_->is_bpe()),
      model_key_(rand()) {
  for (auto idx = 0; idx < alphabet_->labels().size(); idx++) {
    idx2vocab_[idx] = alphabet_->labels().at(idx);
  }
  if (language_model.has_value()) {
    model_container_[model_key_] = language_model.value();
  }
}
std::vector<OutputBeam> BeamSearchDecoderCTC::decode_logits(
    const Eigen::MatrixXf &logits, int beam_width, float beam_prune_logp,
    float token_min_logp, bool prune_history, HotWordScorerPtr hotword_scorer,
    std::optional<AbstractLMStatePtr> lm_start_state) {
  const auto language_model_it = model_container_.find(model_key_);
  LMScoreCache cached_lm_scores;
  if (language_model_it != model_container_.end()) {
    AbstractLMStatePtr start_state =
        lm_start_state.has_value()
            ? lm_start_state.value()
            : language_model_it->second->get_start_state();
    cached_lm_scores.insert(
        std::make_pair(std::make_pair(std::string(""), false),
                       std::make_tuple(0.0, 0.0, start_state)));
  }
  std::unordered_map<std::string, float> cached_p_lm_scores;
  std::vector<Beam> beams{EMPTY_START_BEAM};
  beams = partial_decode_logits(logits, beams, beam_width, beam_prune_logp,
                                token_min_logp, prune_history, hotword_scorer,
                                cached_lm_scores, cached_p_lm_scores);
  // printf("after decode logit\n");
  // for (const auto &b : beams) {
  //   std::cout << "beam: " << b << std::endl;
  // }
  const std::vector<LMBeam> trimmed_beams =
      finalize_beams(beams, beam_width, beam_prune_logp, hotword_scorer,
                     cached_lm_scores, cached_p_lm_scores, true, true);
  // printf("after finalize beams\n");
  std::vector<OutputBeam> output_beams;
  std::transform(
      trimmed_beams.cbegin(), trimmed_beams.cend(),
      std::back_inserter(output_beams),
      [&cached_lm_scores](const LMBeam &lm_beam) {
        const auto last_lm_state_it =
            cached_lm_scores.find(std::make_pair(lm_beam.text_, true));
        const std::optional<AbstractLMStatePtr> last_lm_state =
            last_lm_state_it != cached_lm_scores.end()
                ? std::optional<AbstractLMStatePtr>(
                      std::get<2>(last_lm_state_it->second))
                : std::nullopt;
        std::vector<WordFrames> text_frames;
        std::vector<std::string> lm_beam_text_split;
        boost::algorithm::split(lm_beam_text_split, lm_beam.text_,
                                boost::is_any_of(" "));
        // TODO replace with zip
        for (auto i = 0; i < std::min(lm_beam_text_split.size(),
                                      lm_beam.text_frames_.size());
             i++) {
          text_frames.emplace_back(lm_beam_text_split.at(i),
                                   lm_beam.text_frames_.at(i));
        }
        return OutputBeam{normalize_whitespace(lm_beam.text_), last_lm_state,
                          text_frames, lm_beam.logit_score_, lm_beam.lm_score_};
      });
  return output_beams;
}

std::vector<LMBeam> BeamSearchDecoderCTC::get_lm_beam(
    const std::vector<Beam> &beams, const HotWordScorerPtr hotword_scorer,
    LMScoreCache &cached_lm_scores,
    std::unordered_map<std::string, float> &cached_partial_token_scores,
    bool is_eos) const {
  const auto language_model_it = model_container_.find(model_key_);
  std::vector<LMBeam> new_beams;
  if (language_model_it == model_container_.end()) {
    for (const auto &beam : beams) {
      const auto new_text = merge_token(beam.text_, beam.next_word_);
      const auto lm_hw_score =
          beam.logit_score_ + hotword_scorer->score(new_text) +
          hotword_scorer->score_partial_token(beam.partial_word_);
      new_beams.emplace_back(LMBeam{
          new_text, "", beam.partial_word_, beam.last_char_, beam.text_frames_,
          beam.partial_frames_, beam.logit_score_, lm_hw_score});
    }
    return new_beams;
  }
  for (const auto &beam : beams) {
    const auto new_text = merge_token(beam.text_, beam.next_word_);
    const auto cache_key = std::make_pair(new_text, is_eos);
    float lm_score;
    if (cached_lm_scores.find(cache_key) == cached_lm_scores.end()) {
      //
      auto [_, prev_raw_lm_score, start_state] =
          cached_lm_scores[std::make_pair(beam.text_, false)];
      const auto [score, end_state] = language_model_it->second->score(
          start_state, beam.next_word_, is_eos);
      // printf("text [%s] next words [%s] score [%f]\n", beam.text_.c_str(),
      //  beam.next_word_.c_str(), score);
      const auto raw_lm_score = prev_raw_lm_score + score;
      const auto lm_hw_score = raw_lm_score + hotword_scorer->score(new_text);
      lm_score = lm_hw_score;
      cached_lm_scores[cache_key] =
          std::make_tuple(lm_hw_score, raw_lm_score, end_state);
    }
    const auto &word_part = beam.partial_word_;
    if (!word_part.empty()) {
      if (cached_partial_token_scores.count(word_part) == 0) {
        if (hotword_scorer->contains(word_part)) {
          cached_partial_token_scores[word_part] =
              hotword_scorer->score_partial_token(word_part);
        } else {
          cached_partial_token_scores[word_part] =
              language_model_it->second->score_partial_token(word_part);
        }
      }
    }
    new_beams.emplace_back(LMBeam{
        new_text, "", word_part, beam.last_char_, beam.text_frames_,
        beam.partial_frames_, beam.logit_score_, beam.logit_score_ + lm_score});
  }
  return new_beams;
}

std::vector<Beam> BeamSearchDecoderCTC::partial_decode_logits(
    const Eigen::MatrixXf &logits, std::vector<Beam> &beams, int beam_width,
    float beam_prune_logp, float token_min_logp, bool prune_history,
    const HotWordScorerPtr hotword_scorer, LMScoreCache &cached_lm_scores,
    std::unordered_map<std::string, float> &cached_p_lm_scores,
    int processed_frames) const {
  auto force_next_break = false;
  std::unordered_set<size_t> idx_list;
  // printf("partial decode logit ");
  // for (const auto &bm : beams) {
  //   std::cout << bm;
  // }
  for (auto frame_idx = processed_frames;
       frame_idx - processed_frames < logits.rows(); frame_idx++) {
    const auto col_idx = frame_idx - processed_frames;
    const auto &logit_col = logits.row(col_idx);
    unsigned int max_idx;
    logit_col.maxCoeff(&max_idx);
    idx_list.clear();
    idx_list.insert(max_idx);
    // std::for_each(logit_col.begin(), logit_col.end(),
    //               [&idx_list, &token_min_logp](const float &item) {
    //                 if (item >= token_min_logp) {
    //                   idx_list.insert(item);
    //                 }
    //               });
    for (auto i = 0; i < logit_col.size(); i++) {
      if (logit_col[i] >= token_min_logp) {
        idx_list.insert(i);
      }
    }
    std::stringstream ss;
    for (const auto &x : idx_list) {
      ss << x << " ";
    }
    // printf("idx list [%s] max idx [%d]\n", ss.str().c_str(), (int)max_idx);
    std::vector<Beam> new_beams;
    for (const auto &idx_char : idx_list) {
      const auto p_char = logit_col[idx_char];
      const auto chr = idx2vocab_.at(idx_char);
      for (const auto &beam : beams) {
        // if only blank token or same token
        if (chr == "" || beam.last_char_ == chr) {
          int new_end_frame;
          if (chr == "") {
            new_end_frame = beam.partial_frames_.first;
          } else {
            new_end_frame = frame_idx + 1;
          }
          const auto new_part_frames =
              chr == ""
                  ? beam.partial_frames_
                  : std::make_pair(beam.partial_frames_.first, new_end_frame);
          new_beams.push_back(Beam{
              beam.text_, beam.next_word_, beam.partial_word_, chr,
              beam.text_frames_, new_part_frames, beam.logit_score_ + p_char});
        }
        // if bpe and leading space char
        else if (is_bpe_ && (chr.find(BPE_TOKEN) != std::string::npos ||
                             force_next_break)) {
          force_next_break = false;
          auto clean_char = chr;
          const auto bpe_tok_pos = chr.find(BPE_TOKEN);
          if (bpe_tok_pos != std::string::npos && bpe_tok_pos == 0) {
            clean_char.erase(0);
          }
          if (bpe_tok_pos != std::string::npos &&
              bpe_tok_pos == chr.size() - 1) {
            clean_char.erase(clean_char.size() - 1);
            force_next_break = true;
          }
          const auto new_frame_list =
              beam.partial_word_ == "" ? beam.text_frames_ : [&beam]() {
                std::vector<Frames> new_text_frame = beam.text_frames_;
                new_text_frame.push_back(beam.partial_frames_);
                return new_text_frame;
              }();
          new_beams.push_back(Beam{beam.text_, beam.partial_word_, clean_char,
                                   chr, new_frame_list,
                                   std::make_pair(frame_idx, frame_idx + 1),
                                   beam.logit_score_ + p_char});
        }
        // if not bpe and space char
        else if (!is_bpe_ && chr == " ") {
          const auto new_frame_list =
              beam.partial_word_ == "" ? beam.text_frames_ : [&beam]() {
                std::vector<Frames> new_text_frame = beam.text_frames_;
                new_text_frame.push_back(beam.partial_frames_);
                return new_text_frame;
              }();
          new_beams.push_back(Beam{beam.text_, beam.partial_word_, "", chr,
                                   new_frame_list, NULL_FRAMES,
                                   beam.logit_score_ + p_char});
        }
        // general update of continuing token without space
        else {
          const auto new_part_frames =
              (beam.partial_frames_.first < 0)
                  ? (std::make_pair(frame_idx, frame_idx + 1))
                  : (std::make_pair(beam.partial_frames_.first, frame_idx + 1));
          new_beams.push_back(Beam{
              beam.text_, beam.next_word_, beam.partial_word_ + chr, chr,
              beam.text_frames_, new_part_frames, beam.logit_score_ + p_char});
        }
      }
    }
    // std::cout << "xxx new beams ";
    // for (const auto &bm : new_beams) {
    //   std::cout << bm;
    // }
    // lm scoring and beam pruning
    new_beams = merge_beams(new_beams);
    // std::cout << "xxx merge beam ";
    // for (const auto &bm : new_beams) {
    //   std::cout << bm;
    // }
    auto scored_beams = get_lm_beam(new_beams, hotword_scorer, cached_lm_scores,
                                    cached_p_lm_scores);
    // printf("xxx lm beams ");
    // for (const auto &lmb : scored_beams) {
    //   std::cout << lmb;
    // }
    // remove beam outliers
    const auto max_score_it = std::max_element(
        scored_beams.cbegin(), scored_beams.cend(),
        [](const LMBeam &left, const LMBeam &right) {
          return static_cast<const LMBeam *>(&left)->lm_score_ <
                 static_cast<const LMBeam *>(&right)->lm_score_;
        });
    const auto max_score =
        static_cast<const LMBeam *>(&*max_score_it)->lm_score_;
    const auto score_thresh = max_score + beam_prune_logp;
    scored_beams.erase(
        std::remove_if(
            scored_beams.begin(), scored_beams.end(),
            [&score_thresh](const Beam &item) {
              return (static_cast<const LMBeam *>(&item)->lm_score_) <
                     score_thresh;
            }),
        scored_beams.end());
    const auto trimmed_beams = sort_and_trim_beams(scored_beams, beam_width);
    if (prune_history) {
      const auto language_model_it = model_container_.find(model_key_);
      const auto lm_order = language_model_it == model_container_.end()
                                ? 1
                                : language_model_it->second->order();
      beams = do_prune_history(trimmed_beams, lm_order);
    } else {
      beams.clear();
      std::transform(
          trimmed_beams.begin(), trimmed_beams.end(), std::back_inserter(beams),
          [](const auto &lmbeam) { return Beam::from_lm_beam(lmbeam); });
    }
  }
  return beams;
}

std::string BeamSearchDecoderCTC::decode(
    const Eigen::MatrixXf &logits, int beam_width, float beam_prune_logp,
    float token_min_logp, bool prune_history,
    const std::unordered_set<std::string> &hotwords, float hotword_weight,
    std::optional<AbstractLMState> lm_start_state) {
  auto logit_copy = logits;
  const auto decoded_beams = this->decode_beams(
      logit_copy, beam_width, beam_prune_logp, token_min_logp, true, hotwords,
      hotword_weight, lm_start_state);
  return decoded_beams.at(0).text_;
}

std::vector<OutputBeam> BeamSearchDecoderCTC::decode_beams(
    Eigen::MatrixXf &logits, int beam_width, float beam_prune_logp,
    float token_min_logp, bool prune_history,
    const std::unordered_set<std::string> &hotwords, float hotword_weight,
    std::optional<AbstractLMState> lm_start_state) {
  check_logits_dimension(logits);
  const auto hotword_scorer =
      HotWordScorer::build_scorer(hotwords, hotword_weight);
  // std::cout << "input logits\n" << logits << std::endl;
  // printf("built hotword scorer\n");
  if (std::abs((logits.rowwise().sum()).mean() - 1.0) <
      std::numeric_limits<float>::epsilon()) {
    logits = logits.cwiseMin(1).cwiseMax(MIN_TOKEN_CLIP_P).array().log();
  } else {
    Eigen::MatrixXf temp;
    EMatrixLogSoftmax<1>(logits, temp);
    logits = temp.cwiseMin(0).cwiseMax(std::log(MIN_TOKEN_CLIP_P));
  }
  // std::cout << "logits\n" << logits << std::endl;
  return decode_logits(logits, beam_width, beam_prune_logp, token_min_logp,
                       prune_history, hotword_scorer);
}

std::vector<LMBeam> BeamSearchDecoderCTC::finalize_beams(
    const std::vector<Beam> &beams, int beam_width, float beam_prune_logp,
    HotWordScorerPtr hotword_scorer, LMScoreCache &cached_lm_scores,
    std::unordered_map<std::string, float> &cached_p_lm_scores,
    bool force_next_word, bool is_end) {
  std::vector<Beam> new_beams;
  if (force_next_word || is_end) {
    for (const auto &beam : beams) {
      const auto new_token_times =
          beam.partial_word_ == "" ? beam.text_frames_ : [&beam]() {
            auto new_frames = beam.text_frames_;
            new_frames.push_back(beam.partial_frames_);
            return new_frames;
          }();
      new_beams.push_back(Beam{beam.text_, beam.partial_word_, "", std::nullopt,
                               new_token_times, std::make_pair(-1, -1),
                               beam.logit_score_});
    }
    new_beams = merge_beams(new_beams);
  } else {
    new_beams = beams;
  }
  auto scored_beams = get_lm_beam(new_beams, hotword_scorer, cached_lm_scores,
                                  cached_p_lm_scores);
  const auto max_score_it =
      std::max_element(scored_beams.cbegin(), scored_beams.cend(),
                       [](const LMBeam &left, const LMBeam &right) {
                         return static_cast<const LMBeam *>(&left)->lm_score_ <
                                static_cast<const LMBeam *>(&right)->lm_score_;
                       });
  const auto max_score = static_cast<const LMBeam *>(&*max_score_it)->lm_score_;

  const auto score_thresh = max_score + beam_prune_logp;
  scored_beams.erase(
      std::remove_if(scored_beams.begin(), scored_beams.end(),
                     [&score_thresh](const Beam &item) {
                       return (static_cast<const LMBeam *>(&item)->lm_score_) <
                              score_thresh;
                     }),
      scored_beams.end());
  return sort_and_trim_beams(scored_beams, beam_width);
}

} // namespace pyctcdecode