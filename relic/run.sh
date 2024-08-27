# Change these values

CONFIG_NAME=baseline/relic_HSSD

WB_ENTITY=
WB_JOB_ID="relic_ExtObjNav"
WB_PROJECT_NAME="relic"

DATA_DIR="exp_data"

N_GPUS=1

################################ EVAL PARAMS ##################################


EVAL_N_STEPS=8200
EVAL_N_DEMOS=-1
EVAL_FIX_TARGET_IN_TRIAL=False
EVAL_MAX_NUM_START_POS=-1
EVAL_MAX_NUM_EPISODES=-1
CKPT_NAME=latest.pth # Or ckpt.#.pth where # is the checkpoint number

###############################################################################
CHECKPOINT_FOLDER=$DATA_DIR/checkpoints/$WB_PROJECT_NAME/$WB_JOB_ID/
VIDEO_DIR=$DATA_DIR/vids/$WB_PROJECT_NAME/$WB_JOB_ID/
LOG_FILE=$DATA_DIR/logs/$WB_PROJECT_NAME/$WB_JOB_ID.log
EVAL_DATA_DIR=$DATA_DIR/eval_data/$WB_PROJECT_NAME/$WB_JOB_ID/$CKPT_NAME/

IS_EVAL=${1:-0}

if [ "$IS_EVAL" = "--eval" ] ; then
        echo "here $IS_EVAL"
        EVAL_ARGS=(
                habitat_baselines.eval_ckpt_path_dir=$DATA_DIR/checkpoints/$WB_PROJECT_NAME/$WB_JOB_ID/$CKPT_NAME \
                habitat_baselines.evaluate=True
                habitat_baselines.eval.video_option="[]"
                habitat_baselines.writer_type="tb"
                +habitat_baselines.evaluation_config.n_steps=$EVAL_N_STEPS
                +habitat_baselines.evaluation_config.n_demos=$EVAL_N_DEMOS
                +habitat_baselines.evaluation_config.fix_target_same_episode=$EVAL_FIX_TARGET_IN_TRIAL
                +habitat_baselines.evaluation_config.max_num_start_pos=$EVAL_MAX_NUM_START_POS
                +habitat_baselines.evaluation_config.max_n_eps=$EVAL_MAX_NUM_EPISODES
        )

else
        EVAL_ARGS=()
fi
echo ${EVAL_ARGS[@]}

###############################################################################

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

set -x

python run.py --config-name $CONFIG_NAME \
        habitat_baselines.wb.entity=$WB_ENTITY \
        habitat_baselines.wb.run_name=$WB_JOB_ID \
        habitat_baselines.wb.project_name=$WB_PROJECT_NAME \
        habitat_baselines.checkpoint_folder=$CHECKPOINT_FOLDER \
        habitat_baselines.video_dir=$VIDEO_DIR \
        habitat_baselines.log_file=$LOG_FILE \
        habitat_baselines.eval_data_dir=$EVAL_DATA_DIR \
        habitat_baselines.writer_type=wb \
        ${EVAL_ARGS[@]}
