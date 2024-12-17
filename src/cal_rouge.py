import argparse
from rouge_score import rouge_scorer

# Thiết lập đối số dòng lệnh
def parse_args():
    parser = argparse.ArgumentParser(description="Tính điểm Rouge giữa hai file văn bản.")
    parser.add_argument('--file1', type=str, help="Đường dẫn đến file đầu tiên")
    parser.add_argument('--file2', type=str, help="Đường dẫn đến file thứ hai")
    return parser.parse_args()

# Hàm chính
def main():
    # Nhận các đối số từ dòng lệnh
    args = parse_args()

    # Đọc nội dung của hai file .txt
    with open(args.file1, 'r', encoding='utf-8') as f1:
        text1 = f1.read()

    with open(args.file2, 'r', encoding='utf-8') as f2:
        text2 = f2.read()

    # Tạo một đối tượng rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Tính điểm Rouge
    scores = scorer.score(text1, text2)

    # In ra kết quả
    print(f"Rouge-1: {scores['rouge1']}")
    print(f"Rouge-2: {scores['rouge2']}")
    print(f"Rouge-L: {scores['rougeL']}")

# Gọi hàm chính
if __name__ == '__main__':
    main()

