DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)

#declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'malayalam_male_english' 'assamese_female_english' 'manipuri_female_english' 'tamil_male_english')

declare -a accents=('kannada_male_english')
for seed in 2
do
  for accent in "${accents[@]}"
  do
    echo $accent seed $seed
    python3 -u error_model_sampling.py \
      --selection_json_file=$DATA/$accent/manifests/selection.json \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --error_model_weights=$PRETRAINED_CKPTS/error_models/$accent/seed_"$seed"/best/weights.pkl \
      --random_json_path=$DATA/$accent/manifests/train/random \
      --output_json_path=$DATA/$accent/manifests/train/error_model \
      --exp_id=$seed
  echo
  done
done