# Môn học: Quản lý dự án HTTT - 2425I_INT2045E_53

Github của môn quản lý dự án HTTT của Nhóm 4.  
**Bài toán:** Tóm tắt đa phương tiện cho văn bản và hình ảnh (Multimodal Summarization for Texts and Images).  
**Giáo viên:** Nguyễn Ngọc Hóa, Nguyễn Đức Anh  

## Thành viên
- Đoàn Văn Nguyên (Nhóm trưởng)
- Nguyễn Hải Nam
- Vũ Đại Minh
- Hoàng Quốc Đạt
- Hà Nguyễn Anh Sơn  

## Link Overleaf
[Overleaf Project](https://www.overleaf.com/read/ssmjxfvzrhcz#aacc51)

---

## Hướng dẫn chạy mã nguồn OpenNMT trên MSMO

### 1. Trích xuất dữ liệu
```bash
python data_extraction.py -p /mnt/disks/local_ssd/nhom4/data2/article --type train
# Output:
# train_title.txt, train_document.txt, train_doc_name.txt

python data_extraction.py -p /mnt/disks/local_ssd/nhom4/test_data/article --type test
# Output:
# test_title.txt, test_document.txt, test_doc_name.txt
```

### 2. Làm sạch dữ liệu
```bash
python clean_data.py --type train
# Output:
# src_train.txt, tgt_train.txt

python clean_data.py --type test
# Output:
# src_test.txt, tgt_test.txt
```

### 3. Rút gọn dữ liệu
```bash
python data_truncation.py --type train
# Output:
# ./truncated_docs/t_src_train.txt
# ./truncated_docs/t_tgt_train.txt

python data_truncation.py --type test
# Output:
# ./truncated_docs/t_src_test.txt
# ./truncated_docs/t_tgt_test.txt
```

### 4. Làm sạch chú thích hình ảnh
```bash
python clean_captions.py -i /mnt/disks/local_ssd/nhom4/data2/img/ -c /mnt/disks/local_ssd/nhom4/data2/caption/ --type train
# Output:
# Train_Image_Names.txt, Train_Image_Captions.txt

python clean_captions.py -i /mnt/disks/local_ssd/nhom4/test_data/img/ -c /mnt/disks/local_ssd/nhom4/test_data/caption/ --type test
# Output:
# Test_Image_Names.txt, Test_Image_Captions.txt
```

### 5. Xử lý trước dữ liệu OpenNMT
```bash
cd ../OpenNMT-py/

python preprocess.py -train_src ../src/truncated_docs/t_src_train.txt \
                     -train_tgt ../src/truncated_docs/t_tgt_train.txt \
                     -save_data data/data \
                     --src_seq_length 110 \
                     --src_seq_length_trunc 110 \
                     --tgt_seq_length 26 \
                     --tgt_seq_length_trunc 26 \
                     --src_vocab_size 230000 \
                     --tgt_vocab_size 85000 \
                     -overwrite
```

### 6. Huấn luyện mô hình OpenNMT
```bash
python train.py -data data/data -save_model ./saved_model \
           --valid_steps 1000 --valid_batch_size 128 --save_checkpoint_steps 1000 \
           -global_attention mlp \
           -word_vec_size 128 \
           -rnn_size 512 \
           -layers 1 \
           -encoder_type brnn \
           -train_steps 50000 \
           -max_grad_norm 2 \
           -dropout 0.4 \
           -batch_size 128 \
           -optim adagrad \
           -learning_rate 0.15 \
           -adagrad_accumulator_init 0.1 \
           -gpu_ranks 0 
```
### 7. Dịch văn bản với mô hình huấn luyện
```bash
python translate.py -model saved_model_step_50000.pt -src ../src/truncated_docs/t_src_test.txt -output ./pred_mlp.txt -verbose \
                    --batch_size 512 --gpu 0
```
