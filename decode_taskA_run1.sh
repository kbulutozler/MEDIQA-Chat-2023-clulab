gdown --id 1-_iV-XZlV6Oj46RKYhKoireZK-T8feXM -O header_model.zip
gdown --id 1uyEkGqgI0HZwIcOYhYS8GNg93P2tQygw -O summary_model.zip
mkdir header_model
unzip header_model.zip -d header_model
mkdir summary_model
unzip summary_model.zip -d summary_model

python decode_taskA_run1.py -t $0

