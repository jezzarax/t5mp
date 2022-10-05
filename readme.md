Major parts of the code are not mine, just adapted from https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling

```
ptlm train-model \
 --output_dir="./t5mumo" \
 --model_type="t5" \
 --config_name="./t5mumo" \
 --tokenizer_name="./t5mumo" \
 --dataset_name="wikitext" \
 --dataset_config_name="wikitext-103-v1" \
 --max_seq_length="512" \
 --per_device_train_batch_size="32" \
 --per_device_eval_batch_size="32" \
 --adafactor \
 --learning_rate="0.005" \
 --weight_decay="0.001" \
 --warmup_steps="2000" \
 --overwrite_output_dir \
 --logging_steps="500" \
 --save_steps="10000" \
 --eval_steps="2500"
```

## some notes for machine init

```
sudo apt update
sudo apt install htop tmux vim wget curl git build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev zsh ca-certificates gnupg lsb-release
sudo /opt/deeplearning/install-driver.sh
curl https://pyenv.run | bash
echo 'export PATH="/home/alexeykuntsevich/.local/bin:$PATH"' | tee -a ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' | tee -a ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' | tee -a ~/.bashrc
echo 'eval "$(pyenv init -)"' | tee -a ~/.bashrc
pyenv install 3.8.14
pyenv rehash
pyenv global 3.8.14
vim ~/.ssh/id_rsa
chmod 400 ~/.ssh/id_rsa
curl -sSL https://install.python-poetry.org | python -
git clone git@github.com:jezzarax/t5mp.git
cd t5mp
poetry env use 3.8
poetry shell
poetry install
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```