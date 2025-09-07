# Automatic Model Pairing System

## 🚀 **Fully Automated Model Discovery**

The system now automatically discovers and pairs models from your directory structure without any hardcoding!

## **How It Works**

### **1. Directory Structure**
```
models/
├── base/
│   ├── Qwen_Qwen2.5-1.5B/           # Your base model
│   ├── Qwen2.5-1.5B/                # Another base model
│   └── unsloth_mistral-7b.../       # Any other base model
└── finetuned/
    ├── Qwen_Qwen2.5-1.5B-Instruct/  # Instruction-tuned
    ├── Qwen_Qwen2.5-1.5B-Chat/      # Chat-tuned
    ├── Qwen2.5-1.5B-Instruct/       # Another variant
    └── Any_Other_Model/              # Any other fine-tuned model
```

### **2. Intelligent Pairing Logic**
The system uses **4 strategies** to automatically pair models:

#### **Strategy 1: Exact Core Name Matching**
- `Qwen_Qwen2.5-1.5B` pairs with `Qwen_Qwen2.5-1.5B-Instruct`
- Removes suffixes like `-base`, `-instruct`, `-chat` for matching

#### **Strategy 2: Name Inclusion**
- If a fine-tuned model name contains the base model's core name
- `Qwen2.5-1.5B` pairs with `Qwen2.5-1.5B-Instruct`

#### **Strategy 3: Token Analysis**
- Breaks model names into meaningful tokens
- Pairs models with 2+ common significant tokens
- Example: `qwen`, `2.5`, `1.5b` tokens

#### **Strategy 4: Model Family + Size**
- Matches by model family (`qwen`, `llama`, `mistral`) and size (`1.5b`, `7b`)
- Ensures models from same family and size are paired

### **3. Multiple Pairings Support**
**One base model can have multiple fine-tuned versions:**

For base model `Qwen_Qwen2.5-1.5B`, you'll get:
- `Qwen_Qwen2.5-1.5B` ↔️ `Qwen_Qwen2.5-1.5B-Instruct`
- `Qwen_Qwen2.5-1.5B` ↔️ `Qwen_Qwen2.5-1.5B-Chat`

### **4. What You'll See in the Dropdown**
```
📊 Qwen_Qwen2.5-1.5B ↔️ Qwen_Qwen2.5-1.5B-Instruct (3.0GB)
📊 Qwen_Qwen2.5-1.5B ↔️ Qwen_Qwen2.5-1.5B-Chat (3.0GB)  
📊 Qwen2.5-1.5B ↔️ Qwen2.5-1.5B-Instruct (2.8GB)
```

## **Your Workflow**

### **Step 1: Download Models**
Place your downloaded models in the correct directories:
```bash
# Base model
models/base/Your_Base_Model_Name/

# Fine-tuned models (any names that include the base model)
models/finetuned/Your_Base_Model_Name-Instruct/
models/finetuned/Your_Base_Model_Name-Chat/
models/finetuned/Your_Base_Model_Name-Custom/
```

### **Step 2: Start the Application**
```bash
python app.py
```

### **Step 3: Auto-Discovery Happens**
- System scans both directories
- Applies intelligent pairing logic
- Creates all possible combinations
- Shows them in the selection dropdown

### **Step 4: Select and Compare**
- Choose any pair from the dropdown
- System loads both models
- Start comparing responses side-by-side!

## **Supported Naming Patterns**

### **Base Models:**
- `ModelName`
- `ModelName-base`
- `ModelName_base`
- `Prefix_ModelName`

### **Fine-tuned Models:**
- `ModelName-instruct`
- `ModelName-chat`
- `ModelName-sft`
- `ModelName_custom`
- Any name containing the base model name

## **Features**

### ✅ **Zero Configuration**
- No hardcoding required
- No manual pair definitions needed
- Just drop models in directories

### ✅ **Flexible Naming**
- Handles any naming convention
- Smart pattern recognition
- Prefix/suffix agnostic

### ✅ **Multiple Variants**
- One base → many fine-tuned models
- All combinations available
- Easy selection interface

### ✅ **Research Ready**
- All comparisons logged automatically
- Performance metrics tracked
- CSV export for analysis

## **Example: Qwen Models**

With this structure:
```
models/
├── base/
│   └── Qwen_Qwen2.5-1.5B/
└── finetuned/
    ├── Qwen_Qwen2.5-1.5B-Instruct/
    └── Qwen_Qwen2.5-1.5B-Chat/
```

You automatically get **2 comparison options**:
1. **Base vs Instruct** - Compare base model with instruction-tuned version
2. **Base vs Chat** - Compare base model with chat-optimized version

## **Demo Mode**
Even without real models, the system provides:
- Mock Qwen model pairs for testing
- Realistic comparison responses
- Full interface functionality
- Research logging demonstration

**The system is now completely flexible and will automatically detect any model pairing you want to test!** 🎯
