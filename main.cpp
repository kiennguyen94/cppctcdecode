#include "alphabet.hpp"
#include "constants.hpp"
#include "language_model.hpp"
#include "src/decoder.hpp"
#include <Eigen/Eigen>
#include <fstream>
#include <lm/model.hh>
#include <lm/state.hh>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

static Eigen::Matrix<float, 13, 8> TEST_LOGIT{
    {-3.453877639491068408e+01, 0.000000000000000000e+00,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, 0.000000000000000000e+00,
     -3.453877639491068408e+01, -3.453877639491068408e+01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -7.133498878774647833e-01, -6.733445532637656328e-01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -7.133498878774647833e-01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -6.733445532637656328e-01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -6.733445532637656328e-01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -7.133498878774647833e-01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -6.733445532637656328e-01, -7.133498878774647833e-01},
    {0.000000000000000000e+00, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01},
    {-3.453877639491068408e+01, 0.000000000000000000e+00,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, 0.000000000000000000e+00,
     -3.453877639491068408e+01, -3.453877639491068408e+01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, 0.000000000000000000e+00,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, 0.000000000000000000e+00},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, 0.000000000000000000e+00,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01},
    {-3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     -3.453877639491068408e+01, -3.453877639491068408e+01,
     0.000000000000000000e+00, -3.453877639491068408e+01}};

static std::vector<std::string> SAMPLE_LABELS{" ", "b", "g", "n",
                                              "s", "u", "y", ""};

int main() {
  const pyctcdecode::Beam test_beam{"hi",
                                    "hello",
                                    "world",
                                    std::nullopt,
                                    {pyctcdecode::Frames{1, 2}},
                                    pyctcdecode::Frames{1, 2},
                                    10.0};

  // load test logits
  std::cout << "Matrix:\n" << TEST_LOGIT << std::endl;

  Eigen::MatrixXf output;
  pyctcdecode::EMatrixLogSoftmax<1>(TEST_LOGIT, output);
  std::cout << "Log softmax:\n" << output << std::endl;
  const std::shared_ptr<const lm::ngram::Model> ken_lm =
      std::make_shared<const lm::ngram::Model>(
          "/Volumes/SSD-PGU3/Documents/programming_proj/pyctcdecode/"
          "pyctcdecode/tests/sample_data/bugs_bunny_kenlm.arpa");
  auto start_state = lm::ngram::State();
  auto end_state = lm::ngram::State();
  ken_lm->BeginSentenceWrite(&start_state);
  lm::ngram::State other_start = start_state;
  printf("SCORE %f\n",
         ken_lm->BaseScore(&other_start,
                           ken_lm->BaseVocabulary().Index("bunny"),
                           &end_state));
  printf("index %d\n", ken_lm->BaseVocabulary().Index("bunny"));

  const auto alphabet = pyctcdecode::Alphabet::build_alphabet(SAMPLE_LABELS);
  auto decoder = std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(alphabet);
  std::shared_ptr<pyctcdecode::LanguageModel> language_model;
  std::string text;
  // text = decoder->decode(TEST_LOGIT);
  // printf("decoded [%s]\n", text.c_str());
  // language_model =
  //     std::make_shared<pyctcdecode::LanguageModel>(ken_lm, std::nullopt,
  //     0.0);
  // decoder = std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(alphabet,
  //                                                               language_model);
  // text = decoder->decode(TEST_LOGIT);
  // printf("decoded [%s]\n", text.c_str());

  //   language_model =
  //       std::make_shared<pyctcdecode::LanguageModel>(ken_lm,
  //       std::nullopt, 1.0);
  //   decoder = std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(alphabet,
  //                                                                 language_model);
  //   text = decoder->decode(TEST_LOGIT);
  //   printf("decoded [%s]\n", text.c_str());

  language_model = std::make_shared<pyctcdecode::LanguageModel>(
      ken_lm, std::unordered_set<std::string>{"bunny"}, 1.0,
      pyctcdecode::DEFAULT_BETA, 0.0);
  decoder = std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(alphabet,
                                                                language_model);
  text = decoder->decode(TEST_LOGIT);
  printf("decoded [%s]\n", text.c_str());
}