c &
python image_gen.py --gpu 1 --start 10000 --end 20000 &
python image_gen.py --gpu 2 --start 20000 --end 30000 &
python image_gen.py --gpu 3 --start 30000 --end 40000 &
wait

# chmod +x run.sh
# ./run.sh