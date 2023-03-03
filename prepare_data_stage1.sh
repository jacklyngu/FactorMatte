# Get ready the homographies_raw.txt, mask/01, rgb folder and run this!
python video_completion.py --path $1/rgb --step 1
python video_completion.py --path $1/rgb --step 4
python video_completion.py --path $1/rgb --step 8

mv RAFT_result/$(echo $1 | sed 's/\///g')rgb/*flow* $1

python datasets/confidence.py --dataroot $1 --step 1
python datasets/confidence.py --dataroot $1 --step 4
python datasets/confidence.py --dataroot $1 --step 8

python datasets/homography.py  --homography_path $1/homographies_raw.txt --width $2 --height $3
python data/misc_data_process.py --dataroot $1
