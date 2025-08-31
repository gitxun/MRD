# Mutual Refinement Distillation (MRD) for Multimodal Emotion Recognition

## Introduction

With the rapid advancement of speech emotion recognition, the transition from unimodal to multimodal approaches has become inevitable. However, multimodal methods introduce new challenges, especially classification ambiguity in complex samples compared to unimodal approaches.

**Mutual Refinement Distillation (MRD)** is a novel method designed to address these difficulties by leveraging interactive learning and curriculum strategies for better multimodal emotion recognition.

## Key Features

MRD integrates three major components:

1. **Modal Interaction Calibration**  
   - Enhances the classification accuracy for complex samples by calibrating interactions between modalities.

2. **Interactive Learning Constraints**  
   - Mitigates overfitting through effective interactive learning constraints.

3. **Reverse Curriculum Learning**  
   - Improves model robustness by reversing the traditional curriculum learning order, focusing on more complex samples earlier.

## Pipeline

![Pipeline](https://github.com/user-attachments/assets/c481e063-2deb-4a67-b227-eb4f1b827f25)

## Results

MRD has been evaluated on the following benchmarks:
- **MELD**
- **IEMOCAP**

**Highlights:**
- Outperforms state-of-the-art methods in emotion recognition.
- Achieves a notable **6.07% improvement over the baseline on IEMOCAP**.

## Usage

### 1. Install Requirements

Make sure you have [Python 3.10+](https://www.python.org/downloads/) installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```
---

### 2. Download Data

Download the dataset from the following link:

[Download Data](https://www.dropbox.com/scl/fo/veblbniqjrp3iv3fs3z6p/AEzkNgWqPHHzldBZ0zEzr2Y?rlkey=yhlr653c0vnvaf1krpdkla36u&e=2&dl=0)

After downloading, unzip the data and place it in the `data/` directory of the project. Create this directory if it does not exist.

---

### 3. Quick Start

After installing dependencies and preparing the data, you can quickly start the project with the following commands, depending on the dataset you are using:

#### For IEMOCAP

```bash
./run_emo.sh
```

#### For MELD

```bash
./run_meld.sh
```

> **Tip:** Make sure to adjust the paths if you have placed your data in a different location.  
---

### Notes

- If you encounter issues during installation, please follow the error messages to install any additional required packages.
- For more detailed usage instructions or advanced options (training, testing, etc.), please refer to the project's `README.md` or other documentation files provided in the root directory.

---

For further assistance, please open an issue or contact the project maintainer.

---

## Acknowledgements
Special thanks to the following authors for their contributions through open-source implementations.
- [CMERC](https://github.com/HITSZ-HLT/CMERC)
