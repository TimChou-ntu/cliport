python cliport/eval.py model_task=packing-shapes \
                       eval_task=packing-shapes \
                       agent=ours_trans_two_stream \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       exp_folder=exp_ours_twostreamblip \
                       update_results=True \
                       disp=False
                    #    checkpoint_type=test_best \
