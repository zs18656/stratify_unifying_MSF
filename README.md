
# 📈 Stratify: Unifying Multi-Step Forecasting Strategies

Welcome to the official codebase for **Stratify** — our paper is under review exploring a unified framework for multi-step time series forecasting strategies. Stratify bridges decades of forecasting literature, generalises existing methods, and discovers **novel, higher-performing strategies** — all in one cohesive parameterised space. 🧠📊

🔗 **[Read the paper on arXiv](https://arxiv.org/pdf/2412.20510?)**  
📂 **[BasicTS Benchmark Repository](https://github.com/GestaltCogTeam/BasicTS.git)**  

---

## 🚀 Run an Example

To get started quickly, follow these steps:

```bash
# 1. Download the BasicTS benchmark datasets
#    https://github.com/GestaltCogTeam/BasicTS.git
#    Save as BasicTS_Data

# 2. Set up your environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the example script
python example.py
```

🗂️ Results will be saved in the `stratify_results/` directory.

---

## ✨ What’s in this repo?

- `forecasting_functions.py`: MLP, RNN, LSTM, Transformer, and Random Forest implementations  
- `strategies.py`: Recursive, Direct, DirRec, and all Stratify variants  
- `example.py`: Minimal working example
---

## 🧪 Reproducing Results

Our full experiments span:

- 📊 **18 benchmark datasets**  
- ⏱️ **4 forecast horizons** (10, 20, 40, 80)  
- 🧠 **5 function classes** (MLP, RNN, LSTM, Transformer, RF)  
- 💡 **All existing & novel strategies** unified by Stratify  

---

## 📬 Citation

If you find this framework useful, please consider citing our paper:

```bibtex
@article{green2024stratify,
  title={Stratify: Unifying Multi-Step Forecasting Strategies},
  author={Green, Riku and Stevens, Grant and Abdallah, Zahraa and Silva Filho, Telmo M.},
  journal={arXiv preprint arXiv:2412.20510},
  year={2024}
}
```

---

## 🤝 Contributing

Pull requests and issues are welcome! Whether it’s fixing a typo, adding a strategy variant, or suggesting a better optimiser, we’d love to hear from you. 🙌

---

## 📢 TL;DR

Stratify is a flexible, parameterised framework that:

✅ Unifies existing MSF strategies  
✅ Introduces novel, high-performing methods  
✅ Works across model families and task types  
✅ Makes exploration of the strategy space easy and systematic  
