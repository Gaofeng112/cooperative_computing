#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <regex>
#include <string>
#include <algorithm>
#include <iterator>
#include <zlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

std::string default_bpe() {
    return "./bpe_simple_vocab_16e6.txt.gz";
}

void printByteEncoder(const std::map<std::string, std::string>& byte_encoder) {
    for (const auto& pair : byte_encoder) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
}

std::map<std::string, std::string> bytes_to_unicode() {
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs(bs);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 - 33 + n);
            ++n;
        }
    }
    std::map<std::string, std::string> result;
    for (size_t i = 0; i < bs.size(); ++i) {
        result[std::string(1, static_cast<char>(bs[i]))] = std::string(1, static_cast<char>(cs[i]));
        // std::cout << std::string(1, static_cast<char>(cs[i])) << std::endl;
    }
    return result;
}

std::vector<std::string> bytes_to_unicode_encoder() {
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs(bs);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 - 33 + n);
            ++n;
        }
    }
    std::vector<std::string> result;
    for (size_t i = 0; i < bs.size(); ++i) {
        result.push_back(std::string(1, static_cast<char>(cs[i])));
        // std::cout << std::string(1, static_cast<char>(cs[i])) << std::endl;
    }
    return result;
}

std::vector<std::string> first_division(const std::string &word) {
    std::vector<std::string> result;
    for (size_t i = 0; i < word.size(); ++i) {
        // Check if current character is "</w>"
        if (word[i + 1] == '<' && i + 1 + 3 < word.size() && word.substr(i + 1, 5) == "</w>") {
            // If it's "</w>", treat it as a single string
            result.push_back(word.substr(i, 5));
            i += 4; // Skip "</w>" characters
        } else {
            // Otherwise, treat individual character pairs
            result.push_back(std::string(1, word[i]));
        }
    }
    return result;
}

std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string> &words) {
    std::set<std::pair<std::string, std::string>> pairs;
    std::string prev_char = words[0];
    for (size_t i = 1; i < words.size(); ++i) {
        pairs.emplace(prev_char, words[i]);
        prev_char = words[i]; // Update prev_char
    }
    return pairs;
}

std::string basic_clean(const std::string &text) {
    // Dummy implementation for `ftfy.fix_text` and `html.unescape`
    // In practice, you'd need equivalent libraries or implement these functions.
    return text;
}

std::string whitespace_clean(const std::string &text) {
    std::regex ws_re("\\s+");
    return std::regex_replace(text, ws_re, " ");
}

class SimpleTokenizer {
public:
    SimpleTokenizer(const std::string &bpe_path = default_bpe()) {
        byte_encoder = bytes_to_unicode();
        for (const auto &pair : byte_encoder) {
            byte_decoder[pair.second] = pair.first;
        }
        
        auto byte_encoder_ = bytes_to_unicode_encoder();
        for (const auto &pair : byte_encoder_) {
            vocab.push_back(pair);
        }

        // for (size_t i = 0; i < vocab.size(); ++i) {
        //     std::cout << "vocsb[i]:" << vocab[i] << ", " << i << std::endl;
        // }

        std::vector<std::string> new_vocab;
        new_vocab.reserve(vocab.size()); // 预先分配空间以提高效率

        for (const auto& v : vocab) {
            new_vocab.push_back(v + "</w>");
        }

        // 将新向量中的元素追加到原始向量
        vocab.insert(vocab.end(), new_vocab.begin(), new_vocab.end());

        std::ifstream file(bpe_path, std::ios::in | std::ios::binary);
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content(buffer.str());

        // Decompress the gzipped content
        std::string decompressed;
        z_stream zs;
        memset(&zs, 0, sizeof(zs));
        inflateInit2(&zs, 16 + MAX_WBITS);
        zs.next_in = reinterpret_cast<Bytef *>(const_cast<char*>(content.data()));
        zs.avail_in = content.size();

        int ret;
        char outbuffer[32768];
        do {
            zs.next_out = reinterpret_cast<Bytef *>(outbuffer);
            zs.avail_out = sizeof(outbuffer);
            ret = inflate(&zs, 0);
            if (decompressed.size() < zs.total_out) {
                decompressed.append(outbuffer, zs.total_out - decompressed.size());
            }
        } while (ret == Z_OK);
        inflateEnd(&zs);

        std::istringstream decompressed_stream(decompressed);
        std::string line;
        while (std::getline(decompressed_stream, line)) {
            merges.push_back(line);
        }

        merges = std::vector<std::string>(merges.begin() + 1, merges.begin() + 49152 - 256 - 2 + 1);
        for (const auto &merge : merges) {
            std::istringstream iss(merge);
            std::string s1, s2;
            iss >> s1 >> s2;
            vocab.push_back(s1 + s2);
        }
        vocab.insert(vocab.end(), {"", ""});
        for (size_t i = 0; i < vocab.size(); ++i) {
            // std::cout << "vocsb[i]:" << vocab[i] << ", " << i << std::endl;
            encoder[vocab[i]] = i;
        }

        for (const auto &pair : encoder) {
            decoder[pair.second] = pair.first;
        }
        for (size_t i = 0; i < merges.size(); ++i) {
            std::istringstream iss(merges[i]);
            std::string first_, second_;
      
            // 从istringstream中按空格分隔读取两个字符串
            iss >> first_ >> second_;
      
            // std::cout << "First part: " << first_ << std::endl;
            // std::cout << "Second part: " << second_ << std::endl;
            bpe_ranks[std::make_pair(first_, second_)] = i;
            // std::cout << "merges[i]:" << merges[i] << std::endl;
        }

        cache[""] = "";
        pat = std::regex(
            R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\p{L}\p{N}]+)",
            std::regex_constants::icase);
    }

    std::string bpe(const std::string &token) {
        if (cache.find(token) != cache.end()) {
            return cache[token];
        }
        std::string word = token.substr(0, token.size() - 1) + token.back() + "</w>";
	    // std::cout << "word:" << word << std::endl;
        auto words = first_division(word);
        auto pairs = get_pairs(words);

        // for (const auto& pair_ : pairs) {
        //     std::cout << "{" << pair_.first << ", " << pair_.second << "}" << std::endl;
        // }

        if (pairs.empty()) {
            return token + "</w>";
        }

        while (true) {
            auto bigram = *std::min_element(pairs.begin(), pairs.end(), [&](const std::pair<std::string, std::string> &a, const std::pair<std::string, std::string> &b) {
                return bpe_ranks[a] < bpe_ranks[b];
            });
		    // std::cout << "bigram:{" << bigram.first << ", " << bigram.second << "}" << std::endl;
            if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                break;
            }
            std::vector<std::string> new_word;
            size_t i = 0;
            while (i < words.size()) {
                auto it = std::find(words.begin() + i, words.end(), bigram.first);
                if (it == words.end()) {
                    // 如果找不到, 复制剩余部分到new_word并跳出循环
                    new_word.insert(new_word.end(), words.begin() + i, words.end());
                    break;
                }
                size_t j = std::distance(words.begin(), it);
                // 复制i到j之间的元素到new_word
                new_word.insert(new_word.end(), words.begin() + i, words.begin() + j);
                i = j;
                
                // 检查找到的元素后面是否跟着second
                if (words[i] == bigram.first && i < words.size() - 1 && words[i + 1] == bigram.second) {
                    // 将first+second添加到new_word
                    new_word.push_back(bigram.first + bigram.second);
                    i += 2;
                } else {
                    // 直接将当前元素添加到new_word
                    new_word.push_back(words[i]);
                    i += 1;
                }
            }
            words = new_word;
            if (words.size() == 1) {
                break;
            } else {
                pairs = get_pairs(words);
		
            }
        }
        std::string result;
        for (const auto& s : words) {
            result += s;
        }
        cache[token] = result;
        return result;
    }

    std::vector<int> encode(const std::string &text) {
        std::vector<int> bpe_tokens;
        std::string cleaned_text = whitespace_clean(basic_clean(text));
        std::transform(cleaned_text.begin(), cleaned_text.end(), cleaned_text.begin(), ::tolower);
        std::sregex_token_iterator iter(cleaned_text.begin(), cleaned_text.end(), pat);
        std::sregex_token_iterator end;
        for (; iter != end; ++iter) {
            std::string token = *iter;
            std::string encoded_token;
            // std::cout << "token" << token << std::endl;
            for (unsigned char c : token) {
                encoded_token += byte_encoder[std::string(1, c)];
            }
            std::vector<std::string> bpe_tokens_str;
            std::istringstream bpe_stream(bpe(encoded_token));
            std::copy(std::istream_iterator<std::string>(bpe_stream), std::istream_iterator<std::string>(), std::back_inserter(bpe_tokens_str));
            for (const auto &bpe_token : bpe_tokens_str) {
                bpe_tokens.push_back(encoder[bpe_token]);
            }
        }
        std::vector<int> result = {49406};
        result.insert(result.end(), bpe_tokens.begin(), bpe_tokens.end());
        result.push_back(49407);
        return result;
    }

    std::string decode(const std::vector<int> &tokens) {
        std::string text;
        for (int token : tokens) {
            text += decoder[token];
        }
        std::string result;
        for (char c : text) {
            result += byte_decoder[std::string(1, c)];
        }
        return result;
    }

private:
    std::map<std::string, std::string> byte_encoder;
    std::map<std::string, std::string> byte_decoder;
    std::vector<std::string> merges;
    std::vector<std::string> vocab;
    std::map<std::string, int> encoder;
    std::map<int, std::string> decoder;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    std::map<std::string, std::string> cache;
    std::regex pat;
};

using namespace tflite;

// 函数：将数据保存到 bin 文件
void SaveOutputToBinFile(const std::vector<float>& output_data, const std::string& file_name) {
    std::ofstream out_file(file_name, std::ios::binary);
    if (!out_file) {
        std::cerr << "Unable to open file " << file_name << " write。" << std::endl;
        return;
    }
    out_file.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    out_file.close();
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    // 加载模型
    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile("converted_text_encoder.tflite");
    if (!model) {
        std::cerr << "Unable to load model file." << std::endl;
        return -1;
    }

    // 创建解释器
    ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<Interpreter> interpreter;
    InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Unable to create interpreter." << std::endl;
        return -1;
    }
std::cout << "Distributive tensor" << std::endl;
    // 分配张量
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Cannot assign tensors." << std::endl;
        return -1;
    }
std::cout << "get input tensor" << std::endl;

    SimpleTokenizer tokenizer;
    std::string text = "DSLR photograph of an astronaut riding a horse";
    std::vector<int> encoded = tokenizer.encode(text);
    if (encoded.size() < 77) {
        encoded.resize(77, 49407); // 只会添加新的元素，不会覆盖现有的元素
    }
    for (int token : encoded) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // 获取输入张量
    int* input1 = interpreter->typed_input_tensor<int>(0);
    int* input2 = interpreter->typed_input_tensor<int>(1);
std::cout << "start set input tensor" << std::endl;

    // 将input1和input2的其余元素设置为0
    for (int i = 0; i < 77; ++i) {
        input1[i] = encoded[i];
        std::cout << "input1[i]:" << input1[i] << std::endl; 
    }
    for (int i = 0; i < 77; ++i) {
        input2[i] = i;
        std::cout << "input2[i]:" << input2[i] << std::endl; 
    }

std::cout << "start invoke" << std::endl;
    // 执行推理
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Inference execution failed." << std::endl;
        return -1;
    }

    // 获取输出张量
    const float* output = interpreter->typed_output_tensor<float>(0);
    int output_size = interpreter->tensor(interpreter->outputs()[0])->bytes / sizeof(float);

    // 将输出数据保存到 bin 文件
    std::vector<float> output_data(output, output + output_size);
    SaveOutputToBinFile(output_data, "output.bin");

    std::cout << "The inference is complete and the output is saved to the output.bin file." << std::endl;

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算持续时间，并将其转换为毫秒
    std::chrono::duration<double, std::milli> duration = end - start;

    // 输出持续时间
    std::cout << "Elapsed time: " << duration.count() << " ms" << std::endl;

    return 0;
}

