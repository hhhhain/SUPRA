# while IFS= read -r line
# do
#     echo "当前行内容：$line"
#     CUDA_VISIBLE_DEVICES=0 python main.py --model facebook/opt-125m \
#     --prune_method magnitude --sparsity_type unstructured --sparsity_ratio 0.4 \
#     --save results --save_model "mag_based_opt_12504$line" --sheet_name 'fc1_qkvofc2various_layer1.xlsx'
# done < name_fc1_qkvofc2various_layer1_1.txt


# 定义一个包含所有文本文件名的数组
# txt_files=("name_fc1_various_10total_layer1.txt" "name_fc1_various_15total_layer1.txt" "name_fc1_various_20total_layer1.txt" "name_fc1_various_25total_layer1.txt" "name_fc1_various_30total_layer1.txt" "name_fc1_various_35total_layer1.txt" "name_fc1_various_45total_layer1.txt")
# txt_files=("name_k_various_10total_layer4.txt" "name_k_various_15total_layer4.txt" "name_k_various_20total_layer4.txt" "name_k_various_25total_layer4.txt" "name_k_various_30total_layer4.txt" "name_k_various_35total_layer4.txt" "name_k_various_45total_layer4.txt")



txt_files=("$@")

# 遍历数组中的每个文件名
for txt_file in "${txt_files[@]}"
do
    echo "当前处理的文件：$txt_file"
    # 直接捕获整个需要的字符串部分
    xlsx_part=$(echo "$txt_file" | sed -E 's/name_(k_various_[0-9]+total_layer4)\.txt/\1/')
    # 构建完整的xlsx文件名
    xlsx_file="${xlsx_part}.xlsx"
    echo "对应的xlsx文件名: $xlsx_file"

    while IFS= read -r line
    do
        echo "当前行内容：$line"


        CUDA_VISIBLE_DEVICES=0 python -u main.py --model your_model_path \
        --prune_method wanda --sparsity_type unstructured --sparsity_ratio 0.6 \
        --save results --save_model "mag_based_opt_12504$line" --sheet_name "$xlsx_file"


        # # zero-shot
        # CUDA_VISIBLE_DEVICES=1 python -u main.py --model your_model_path \
        # --prune_method wanda --sparsity_type unstructured --sparsity_ratio 0.4 \
        # --save results --save_model "mag_based_opt_12504$line" --sheet_name "$xlsx_file" --eval_zero_shot


    done < "$txt_file"
done