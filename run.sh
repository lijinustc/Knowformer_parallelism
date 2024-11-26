#FB15k-237
python main.py --seed 42 --accelerator gpu --strategy ddp --precision 32 --devices 2 --max_epochs 20 --checkpoint_save_path ./experiments/fb15k-237/ --data_path ./data/fb15k-237 --batch_size 96 --test_batch_size 96 --num_workers 8 --num_layer 3 --num_qk_layer 2 --num_v_layer 3 --hidden_dim 32 --num_heads 4 --loss_fn bce --adversarial_temperature 0.5 --remove_all --num_negative_sample 8 --learning_rate 5e-3 --optimizer Adam --weight_decay 1e-4  
#python -m debugpy --listen 64921 --wait-for-client main.py --seed 42 --accelerator gpu --strategy ddp --precision 32 --devices 2 --max_epochs 20 --checkpoint_save_path ./experiments/fb15k-237/ --data_path ./data/fb15k-237 --batch_size 96 --test_batch_size 96 --num_workers 8 --num_layer 3 --num_qk_layer 2 --num_v_layer 3 --hidden_dim 32 --num_heads 4 --loss_fn bce --adversarial_temperature 0.5 --remove_all --num_negative_sample 8 --learning_rate 5e-3 --optimizer Adam --weight_decay 1e-4 