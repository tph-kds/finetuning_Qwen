# finetuning_Qwen
### Prerequisites
    Python 3.10 is required or > 3.10.

### Installation and Training Steps
    * 1. mkdir son_llm
    * 2. cd son_llm
    * 3. pip install -r requirements.txt
    * 4. python fine_tuning.py --batch_size=512 --model_type=7 
	(Where model_type = 7 ==> using 7B of Qwen2.5-7B)

### Model Output and Storage
    After training is complete, a folder named 'model' will be generated.
    This folder contains the trained model weights and necessary files.
    All files inside this folder must be kept as they are required for further training, evaluation, or inference.