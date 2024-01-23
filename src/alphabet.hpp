#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyctcdecode {
using Labels = std::vector<std::string>;
class Alphabet;
using AlphabetPtr = std::shared_ptr<const Alphabet>;
class Alphabet {
private:
  bool is_bpe_;
  Labels labels_;
  Alphabet(Labels labels, bool is_bpe);

public:
  Alphabet() = default;
  bool is_bpe() const { return is_bpe_; }
  Labels labels() const { return labels_; }
  static AlphabetPtr build_alphabet(const Labels &);
  std::string dumps() {
    throw std::runtime_error("Alphabet::dumps not implemented");
  }
  static Alphabet loads(const std::string &) {
    throw std::runtime_error("Alphabet::loads not implemented");
  }
};

} // namespace pyctcdecode