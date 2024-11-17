pip install gdown
cd data

mkdir -p Split_Dataset/Data
mkdir -p Split_Dataset/Ground_truth


cd Split_Dataset/Data/
gdown https://drive.google.com/uc?id=1L-4spewLWN7jMzA5uTiBNTHhPl926j64 -O relish_documents.tsv
gdown https://drive.google.com/file/uc?id=1-c-00aJd_ybSL17Jqs22nEKMPd1lz4TZ -O input_train_text_data.tsv
gdown https://drive.google.com/file/uc?id=1uTORTQlIx-IOMIBcj4v1yLFX2XB-klwP -O input_test_text_data.tsv
gdown https://drive.google.com/file/uc?id=1cbHgYKPtocC8Tu5daez34fN0M1_Be6AJ -O input_valid_text_data.tsv

cd ../Ground_truth
gdown  https://drive.google.com/uc?id=1y9T41Faf9Oq2XOtWMD1U9fZe9OHLgLjv -O test.tsv
gdown  https://drive.google.com/uc?id=1R1i74XWzILnlozwCfYItlequKIhMnHmB -O train.tsv
gdown https://drive.google.com/uc?id=1ZupxAdTOWxmKPWlD5FOwEKbkdavt5Zxk -O valid.tsv
