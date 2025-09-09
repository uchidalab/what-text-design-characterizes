# Book Cover Classification

A multimodal deep learning system for book genre classification using cover images with text analysis.

## Overview

This project classifies book genres by analyzing both visual and textual elements from book cover images using a hierarchical Vision Transformer architecture.

## Features

- **Multimodal Learning**: Combines 6 feature types:
  - Semantic: Word2Vec embeddings for text meaning
  - Font Style: ResNet50-extracted font features
  - Character Color: RGB color distribution
  - Background Color: Background color distribution
  - Height: Relative word height information
  - Coordinates: Word position on cover

- **Hierarchical Architecture**: 
  - Word-level: Integrates multiple features per word
  - Title-level: Processes word sequences for final classification

- **Flexible Training**: Configurable feature selection and ablation studies

## Requirements

- Docker
All Python dependencies and environment setup are handled by Docker.

See `docker/README.md` for detailed Docker setup instructions.

## Usage

### Training
```bash
# Basic training
python src/train.py

# Custom configuration
python src/train.py --num_epochs 500 --lr 1e-4 --batch_size 32 --drop_elements semantic
```

### Testing
```bash
# Basic testing
python src/test.py

# Custom configuration  
python src/test.py --drop_elements height coord
```

### Analysis of Test Results
To analyze and visualize the test results, use the Jupyter notebook:
```bash
# For performance analysis and ablation studies
src/test_analysis.ipynb
```

## Dataset Structure and Requirements

### Dataset Download Options

#### Test-Only Dataset (Provided)
- **Size**: ~3GB
- **Content**: Test split data only (5,641 book covers with word detection results)
- **Download**: [dataset.zip](https://drive.google.com/file/d/1LOGsD00v7QT5GJ8NekDNyNnIVP0iIYLF/view?usp=sharing)
- **Purpose**: For model evaluation and testing

#### Pre-trained Models (Provided)
- **Size**: 1.6GB
- **File**: models.zip
- **Content**: Pre-trained model checkpoints for evaluation
- **Download**: [models.zip](https://drive.google.com/file/d/1Iz-2WlOlejmK4WTFfsgQONphwT3GVXDi/view?usp=sharing)
- **Purpose**: For running tests without training from scratch

#### Training Dataset (User Preparation Required)
- **Size**: ~50GB+
- **Content**: All training data
- **Note**: Training data is not provided. Users must prepare their own training dataset following the structure specified below.
- **Purpose**: For training new models

### Dataset Structure
The test-only dataset is provided with the following structure. For training, users need to prepare their own dataset following this exact same structure:

```
dataset/
├── AmazonBookCoverImages/        # Book cover images and metadata (9.9GB)
│   ├── train.csv                 # Training split metadata (99MB)
│   ├── test.csv                  # Test split metadata (11MB)
│   ├── csv/
│   │   └── valid.pickle          # Validation split indices
│   └── genres/                   # Book cover images by category (30 categories)
│       ├── Arts & Photography/
│       ├── Biographies & Memoirs/
│       ├── Business & Money/
│       ├── ... (30 categories total)
│       └── Travel/
│           └── [ASIN].jpg       # Book cover images named by Amazon ID
│
├── word_detection_from_bookCover/  # Word detection results (42GB)
│   └── dataset/
│       ├── [Category Name]/      # Same 30 categories as above
│       │   ├── word/             # Individual word images
│       │   │   └── [book_folder_id]/
│       │   │       └── word_*.jpg
│       │   └── ditection/        # Detection visualization
│       │       └── [book_folder_id].jpg
│       └── ...
│
├── CannotRead/                   # Fallback for unreadable books (185MB)
│   ├── word/                     # Unreadable word images
│   │   └── [book_folder_id]/
│   │       └── word_*.jpg
│   └── ditection/                # Unreadable detection results
│       └── [book_folder_id].jpg
│
├── GoogleNews-vectors-negative300.bin.gz  # Pre-trained Word2Vec (1.6GB)
├── book30-listing-train.csv              # Training book metadata (9.3MB)
└── book30-listing-test.csv               # Test book metadata (1.1MB)
```

### Required Files Details

1. **Word2Vec Model** (1.6GB)
   - Download from: https://code.google.com/archive/p/word2vec/
   - File: `GoogleNews-vectors-negative300.bin.gz`
   - Contains 3 million word vectors trained on Google News

2. **Book Metadata CSVs**
   - Format: CSV with encoding='cp932'
   - Columns: `Amazon ID (ASIN), Filename, Image URL, Title, Author, Category ID, Category`
   - `book30-listing-train.csv`: ~8,700 training books
   - `book30-listing-test.csv`: ~900 test books

3. **Training/Test Split CSVs**
   - Located in `AmazonBookCoverImages/`
   - Required columns:
     - `folder`: Book folder identifier
     - `word`: Detected word text
     - `img_name`: Word image filename
     - `hight`, `width`: Word dimensions
     - `coord_x`, `coord_y`: Word position
     - `book_cover_hight`, `book_cover_width`: Cover dimensions
     - `split`: 'train' or 'test'

4. **Book Cover Images**
   - Format: JPG
   - Resolution: Variable (typically 300-500px width)
   - Organized by genre in `genres/` subdirectories

5. **Word Detection Results**
   - Pre-processed using text detection algorithm
   - Each book's words saved as individual images
   - Required for font style feature extraction

### Optional: Font File
For visualization purposes (not required for training/testing):
1. Download Open Sans from: https://fonts.google.com/specimen/Open+Sans
2. Place `OpenSans-Regular.ttf` in `dataset/`
3. System will use default font if not available

## Models

- **Word-level Model**: Vision Transformer for feature integration
- **Title-level Model**: Vision Transformer for sequence classification
- **Font Model**: Pre-trained ResNet50 for font style extraction
- **Embeddings**: InputEmbed layers for positional information

## Output

- Classification results with top-5 predictions
- Attention visualization data
- Training metrics and loss curves
- Model checkpoints

## Citation

This work is based on the research published in:

```bibtex
@inproceedings{haraguchi2024text,
  title={What text design characterizes book genres?},
  author={Haraguchi, Daichi and Iwana, Brian Kenji and Uchida, Seiichi},
  booktitle={International Workshop on Document Analysis Systems},
  pages={165--181},
  year={2024},
  organization={Springer}
}
```
