export EXP_DIR=./results
export N_STEPS=1000
export SERVER_NAME=a4000
export RUN_NAME=run_1
export LOSS=card_conditional
export TASK=linear_regression
# export TASK=8gauss
export N_SPLITS=20
export N_THREADS=4
export DEVICE_ID=0

export CAT_F_PHI=_cat_f_phi
export MODEL_VERSION_DIR=card_conditional_uci_results/${N_STEPS}steps/nn/${RUN_NAME}_${SERVER_NAME}/f_phi_prior${CAT_F_PHI}
# python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --n_splits ${N_SPLITS} --doc ${TASK} --config configs/${TASK}.yml  #--train_guidance_only
python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test

#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 14 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 15 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 16 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 17 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 18 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 19 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
