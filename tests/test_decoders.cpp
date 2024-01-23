#include "constants.hpp"
#include <optional>
#include <set>
#include <unordered_set>
#define BOOST_TEST_MODULE cppctcdecode
// #include "src/decoder.hpp"
#include "decoder.hpp"
#include <boost/test/included/unit_test.hpp>
#include <lm/model.hh>
namespace {
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
} // namespace

BOOST_AUTO_TEST_CASE(free_test_function)
/* Compare with void free_test_function() */
{
  Eigen::MatrixXf output;
  pyctcdecode::EMatrixLogSoftmax<1>(TEST_LOGIT, output);

  // load kenlm
  const auto ken_lm = std::make_shared<const lm::ngram::Model>(
      "/Volumes/SSD-PGU3/Documents/programming_proj/pyctcdecode/"
      "pyctcdecode/tests/sample_data/bugs_bunny_kenlm.arpa");
  auto start_state = lm::ngram::State();
  auto end_state = lm::ngram::State();
  ken_lm->BeginSentenceWrite(&start_state);
  lm::ngram::State other_start = start_state;
  // build alphabet
  const auto alphabet = pyctcdecode::Alphabet::build_alphabet(SAMPLE_LABELS);

  {
    auto decoder =
        std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(alphabet);
    std::string text = decoder->decode(TEST_LOGIT);
    BOOST_CHECK_EQUAL(text, "bunny bunny");
  }
  {
    auto language_model =
        std::make_shared<pyctcdecode::LanguageModel>(ken_lm, std::nullopt, 1.0);
    auto decoder = std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(
        alphabet, language_model);
    std::string text = decoder->decode(TEST_LOGIT);
    BOOST_CHECK_EQUAL(text, "bugs bunny");
  }
  {
    auto language_model = std::make_shared<pyctcdecode::LanguageModel>(
        ken_lm, std::unordered_set<std::string>(), 1.0);
    auto decoder = std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(
        alphabet, language_model);
    std::string text = decoder->decode(TEST_LOGIT);
    BOOST_CHECK_EQUAL(text, "bugs bunny");
  }
  {
    auto language_model = std::make_shared<pyctcdecode::LanguageModel>(
        ken_lm, std::unordered_set<std::string>{"bunny"}, 1.0,
        pyctcdecode::DEFAULT_BETA, 0.0);
    auto decoder = std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(
        alphabet, language_model);
    std::string text = decoder->decode(TEST_LOGIT);
    BOOST_CHECK_EQUAL(text, "bugs bunny");
  }
  {
    auto language_model = std::make_shared<pyctcdecode::LanguageModel>(
        ken_lm, std::unordered_set<std::string>{"bunny"}, 1.0,
        pyctcdecode::DEFAULT_BETA, -10.0);
    auto decoder = std::make_unique<pyctcdecode::BeamSearchDecoderCTC>(
        alphabet, language_model);
    std::string text = decoder->decode(TEST_LOGIT);
    BOOST_CHECK_EQUAL(text, "bunny bunny");
  }
}
