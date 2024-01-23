#pragma once
#include <math.h>
#include <regex>
#include <string>

#if !defined(PYCTCDECODE_CONST)
#define PYCTCDECODE_CONST 1
namespace pyctcdecode {
const float DEFAULT_ALPHA = 0.5;
const float DEFAULT_BETA = 1.5;
const float DEFAULT_UNK_LOGP_OFFSET = -10.0;
const bool DEFAULT_SCORE_LM_BOUNDARY = true;
const int DEFAULT_BEAM_WIDTH = 100;
const float DEFAULT_HOTWORD_WEIGHT = 10.0;
const float DEFAULT_PRUNE_LOGP = -10.0;
const bool DEFAULT_PRUNE_BEAMS = false;
const float DEFAULT_MIN_TOKEN_LOGP = -5.0;

const int AVG_TOKEN_LEN = 6;
const float MIN_TOKEN_CLIP_P = 1e-15;
const float LOG_BASE_CHANGE_FACTOR = 1.0 / log10(M_E);

const std::string BPE_TOKEN{"▁"};
const std::string BPE_TOKEN_ALT{"##"};
const std::string UNK_TOKEN{"⁇"};
const std::string UNK_BPE_TOKEN{"▁⁇▁"};

const std::regex BLANK_TOKEN_PTN(R"(^[<\[]pad[>\]]$)",
                                 std::regex_constants::icase |
                                     std::regex_constants::ECMAScript);
const std::regex UNK_TOKEN_PTN(R"(^[<\[]unk[>\]]$)",
                               std::regex_constants::icase |
                                   std::regex_constants::ECMAScript);
} // namespace pyctcdecode
#endif