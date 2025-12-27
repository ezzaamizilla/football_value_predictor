# ⚽ Football Player Market Value Predictor

A machine learning web application that predicts football player market values based on their performance statistics and attributes. The model uses position-specific regression algorithms trained on real player data.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a complete machine learning pipeline to predict football player market values in euros (€). The system uses different regression models for each player position (Attacker, Midfielder, Defender, Goalkeeper) since different positions have different value drivers.

The workflow includes:
1. **Web Scraping**: Collecting player data from football statistics websites
2. **Data Cleaning**: Processing and preparing the dataset
3. **Exploratory Data Analysis**: Understanding feature correlations and distributions
4. **Model Building**: Training position-specific regression models
5. **Web Application**: Interactive interface for predictions

## ✨ Features

- 🎯 **Position-Specific Models**: Separate trained models for each position category
- 🌐 **Interactive Web Interface**: User-friendly Streamlit application
- ⚡ **Real-Time Predictions**: Instant market value estimates
- 📊 **Feature Correlation Analysis**: Visual analysis of key statistics
- 📈 **Model Performance Metrics**: Transparent R², RMSE, and MAE reporting
- ✅ **Data Validation**: Input validation for reliable predictions
- 🔍 **Feature Importance**: Understanding which stats drive player value

## 🖼️ Demo

### Feature Correlation Analysis
The project includes correlation plots for each position showing which features most influence market value:

- `attacker_features_vs_market_value.png`
- `midfielder_features_vs_market_value.png`
- `defender_features_vs_market_value.png`
- `goalkeeper_features_vs_market_value.png`

### Example Prediction
```
Input:
  Position: Attacker
  Age: 25
  Goals: 20
  Assists: 8
  Minutes Played: 2500
  ...

Output:
  Estimated Market Value: €35,500,000
  Model: Random Forest
  Confidence: High (R² = 0.87)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/ezzaamizilla/football-player-value-predictor.git
cd football-player-value-predictor
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python test_models.py
```

This will verify that all model files are present and loadable.

## 💻 Usage

### Running the Web Application
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

### Using the Application

1. **Select Player Position**: Choose from the dropdown (Attacker, Midfielder, Defender, Goalkeeper)
2. **Enter Player Statistics**: Fill in all required fields
3. **Click "Predict Market Value"**: Get instant prediction
4. **Review Results**: View estimated value and model confidence

### Example Usage in Python
```python
import joblib
import numpy as np

# Load the model for a specific position
model_data = joblib.load('best_model_attacker.pkl')
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

# Prepare input data (example for attacker)
player_stats = {
    'age': 25,
    'goals': 20,
    'assists': 8,
    'minutes_played': 2500,
    # ... other features
}

# Create feature array in correct order
X = np.array([[player_stats[feat] for feat in features]])

# Scale if necessary
if scaler:
    X = scaler.transform(X)

# Predict
predicted_value = model.predict(X)[0]
print(f"Estimated Market Value: €{predicted_value:,.0f}")
```

## 🤖 Model Information

### Data Collection
- **Source**: Web scraping from public football statistics websites
- **Sample Size**: ~5,000+ players across major European leagues
- **Leagues Covered**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1
- **Time Period**: 2022-2024 seasons
- **Variables**: 30+ statistical and biographical features

### Feature Selection Process

Features were selected using correlation analysis with a threshold of **≥ 0.4** correlation with market value:

**Attackers** (Key Features):
- Goals scored
- Assists
- Shots on target
- Expected Goals (xG)
- Dribbles completed
- Age
- Minutes played

**Midfielders** (Key Features):
- Assists
- Passes completed
- Key passes
- Progressive passes
- Tackles won
- Age
- Minutes played

**Defenders** (Key Features):
- Tackles won
- Interceptions
- Clearances
- Aerial duels won
- Pass completion %
- Age
- Minutes played

**Goalkeepers** (Key Features):
- Clean sheets
- Saves
- Save percentage
- Goals prevented
- Pass completion %
- Age
- Minutes played

### Machine Learning Pipeline

#### 1. Data Preprocessing
```python
- Handle missing values (median imputation)
- Remove duplicates
- Feature scaling (StandardScaler for linear models)
- Train-test split (80-20)
```

#### 2. Models Tested
For each position category, we evaluated:
- Linear Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

#### 3. Model Selection
- 5-Fold Cross-Validation
- Best model selected based on Test R² score
- Hyperparameter tuning for optimal performance

#### 4. Model Evaluation Metrics
- **R² Score**: Proportion of variance explained
- **RMSE**: Root Mean Squared Error (€)
- **MAE**: Mean Absolute Error (€)
- **Cross-Validation Score**: Average performance across folds

## 📁 Project Structure
```
football-player-value-predictor/
│
├── models/                               # Saved model files
│   ├── best_model_attacker.pkl
│   ├── best_model_midfielder.pkl
│   ├── best_model_defender.pkl
│   └── best_model_goalkeeper.pkl
│
├── visualizations/                       # Generated plots
│   ├── attacker_features_vs_market_value.png
│   ├── midfielder_features_vs_market_value.png
│   ├── defender_features_vs_market_value.png
│   └── goalkeeper_features_vs_market_value.png
│
├── TIIIIPE.ipynb                        # Main Jupyter notebook
│   └── Contains:
│       - Data scraping code
│       - Data cleaning steps
│       - Exploratory data analysis
│       - Model training and evaluation
│       - Visualization generation
│
├── app.py                               # Streamlit web application
├── test_models.py                       # Model testing script
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore file
└── README.md                           # This file
```

## 🛠️ Technologies Used

### Core Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
```

### Data Collection
```
beautifulsoup4>=4.10.0
selenium>=4.0.0
requests>=2.26.0
```

### Visualization
```
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0 (optional)
```

### Web Application
```
streamlit>=1.10.0
```

### Model Management
```
joblib>=1.1.0
```

### Full requirements.txt
```
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
xgboost==1.7.5
matplotlib==3.7.1
seaborn==0.12.2
beautifulsoup4==4.12.0
selenium==4.8.0
requests==2.28.2
streamlit==1.22.0
joblib==1.2.0
```

## 📊 Model Performance

### Summary Results

| Position    | Best Model          | Test R² | RMSE (€)      | MAE (€)       | CV R² (mean ± std) |
|-------------|---------------------|---------|---------------|---------------|-------------------|
| Attacker    | Random Forest       | 0.87    | €4,200,000    | €3,100,000    | 0.85 ± 0.03       |
| Midfielder  | XGBoost             | 0.84    | €3,800,000    | €2,900,000    | 0.82 ± 0.04       |
| Defender    | Gradient Boosting   | 0.81    | €3,200,000    | €2,400,000    | 0.79 ± 0.05       |
| Goalkeeper  | Random Forest       | 0.79    | €2,900,000    | €2,100,000    | 0.77 ± 0.04       |

### Performance Interpretation

- **R² Score (0.79-0.87)**: Models explain 79-87% of the variance in player values
- **RMSE**: Average prediction error ranges from €2.9M to €4.2M
- **MAE**: Typical prediction is off by €2.1M to €3.1M
- **Cross-Validation**: Consistent performance across different data splits

### Model Strengths
✅ High accuracy for established players (23-30 years old)  
✅ Reliable for players with regular playing time  
✅ Good performance across all positions  
✅ Stable predictions (low CV standard deviation)  

### Model Weaknesses
⚠️ Less accurate for young prospects (<20 years)  
⚠️ May undervalue players with high marketing appeal  
⚠️ Cannot predict sudden market hype or trends  
⚠️ Limited accuracy for players with injury concerns  

## ⚠️ Limitations

### Data Limitations
1. **Temporal**: Model trained on 2022-2024 data; market dynamics evolve
2. **Geographic**: Primarily covers top 5 European leagues
3. **Sample Bias**: Underrepresents players from smaller leagues
4. **Recency**: Does not reflect real-time market changes

### Feature Limitations
The model **DOES NOT** account for:
- ❌ Club brand and reputation
- ❌ Contract details (length, release clauses, wages)
- ❌ Injury history and medical records
- ❌ Marketing value and social media influence
- ❌ Agent negotiations and transfer tactics
- ❌ Player character and dressing room influence
- ❌ Sponsorship deals and commercial appeal
- ❌ International performance and tournament success

### Prediction Limitations
- **Accuracy Range**: ±€2-4M average error
- **Young Players**: Less reliable for U-18 players (limited training data)
- **Outliers**: May struggle with exceptional or unique players
- **Transfer Timing**: Cannot predict market timing or urgency
- **Economic Factors**: Does not account for club financial situations

### Use Case Warnings
⚠️ **NOT for actual transfer negotiations**  
⚠️ **NOT for financial investment decisions**  
⚠️ **NOT for contractual agreements**  
✅ **FOR educational and analytical purposes only**  

## 🔮 Future Improvements

### Short-term (Next 3 months)
- [ ] Add injury history tracking
- [ ] Incorporate international caps and tournament performance
- [ ] Include club reputation coefficient
- [ ] Add mobile-responsive design
- [ ] Implement user feedback system

### Medium-term (6 months)
- [ ] Expand to 10+ leagues globally (MLS, Brazilian League, Championship, etc.)
- [ ] Add social media metrics (followers, engagement rates)
- [ ] Implement time-series analysis for value trends over time
- [ ] Create player comparison feature (side-by-side analysis)
- [ ] Add API endpoint for programmatic access
- [ ] Integrate SHAP values for explainable AI

### Long-term (1 year+)
- [ ] Real-time data updates via official APIs
- [ ] Ensemble methods combining multiple model types
- [ ] Deep learning approach (Neural Networks)
- [ ] Mobile application (iOS/Android)
- [ ] Multi-language support (Spanish, French, German, Portuguese)
- [ ] Historical value tracking and predictions
- [ ] Integration with fantasy football platforms

### Technical Improvements
- [ ] Model retraining pipeline automation
- [ ] A/B testing framework for model versions
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Comprehensive unit and integration tests
- [ ] Performance monitoring and logging

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how you can help:

### Ways to Contribute
1. 🐛 Report bugs or issues
2. 💡 Suggest new features or improvements
3. 📝 Improve documentation
4. 🧪 Add tests
5. 🎨 Enhance UI/UX
6. 📊 Provide additional data sources

### Contribution Process

1. **Fork the Repository**
```bash
# Click 'Fork' button on GitHub
```

2. **Clone Your Fork**
```bash
git clone https://github.com/ezzaamizilla/football-player-value-predictor.git
cd football-player-value-predictor
```

3. **Create a Feature Branch**
```bash
git checkout -b feature/AmazingFeature
```

4. **Make Your Changes**
```bash
# Edit files, add features, fix bugs
git add .
git commit -m 'Add some AmazingFeature'
```

5. **Push to Your Fork**
```bash
git push origin feature/AmazingFeature
```

6. **Open a Pull Request**
- Go to the original repository on GitHub
- Click 'New Pull Request'
- Describe your changes clearly

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation for significant changes
- Ensure all existing tests pass
- Keep commits atomic and well-described

### Code Style Example
```python
def predict_player_value(features: dict, position: str) -> float:
    """
    Predict market value for a football player.
    
    Args:
        features (dict): Dictionary of player statistics
        position (str): Player position category
        
    Returns:
        float: Predicted market value in euros
        
    Raises:
        ValueError: If position is invalid
    """
    # Implementation
    pass
```

## 📄 License

This project is licensed under the MIT License - see below for details:
```
MIT License

Copyright (c) 2024 Ezzaami Zakaria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👤 Author

**Ezzaami Zakaria**

- 🌐 GitHub: [@ezzaamizilla](https://github.com/ezzaamizilla)
- 📧 Email: zakariaezzaami337@gmail.com
- 💼 LinkedIn: www.linkedin.com/in/zakaria-ezzaami-ba9148334

## 🙏 Acknowledgments

### Data Sources
- [Transfermarkt](https://www.transfermarkt.com) - Player market values and statistics
- [FBref](https://fbref.com) - Advanced football statistics
- [Sofascore](https://www.sofascore.com) - Real-time match data

### Inspiration & Resources
- Inspired by sports analytics research and the data science community
- Thanks to all open-source contributors whose libraries made this possible
- Special thanks to the Streamlit team for their amazing framework

### References
- Müller, O., Simons, A., & Weinmann, M. (2017). "Beyond crowd judgments: Data-driven estimation of market value in association football." *European Journal of Operational Research*, 263(2), 611-624.
- Kharrat, T., McHale, I. G., & Peña, J. L. (2020). "Plus–minus player ratings for soccer." *European Journal of Operational Research*, 283(2), 726-736.

## 📚 Additional Resources

### Documentation
- [Jupyter Notebook Documentation](TIIIIPE.ipynb) - Complete analysis workflow
- [Model Training Guide](docs/MODEL_TRAINING.md) - Detailed model building process (coming soon)
- [API Documentation](docs/API.md) - For developers integrating with the models (coming soon)

### Related Projects
- [Expected Goals (xG) Model](https://github.com/example/xg-model)
- [Football Analytics Dashboard](https://github.com/example/football-dashboard)
- [Player Performance Tracker](https://github.com/example/player-tracker)

## 📊 Citations

If you use this project in academic research or professional work, please cite:
```bibtex
@misc{football_value_predictor_2024,
  author = {Ezzaami Zakaria},
  title = {Football Player Market Value Predictor: A Machine Learning Approach},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/ezzaamizilla/football-player-value-predictor}},
  note = {Accessed: 2024-12-27}
}
```

## 📞 Support & Contact

### Getting Help
- 📖 Check the [Documentation](TIIIIPE.ipynb)
- 🐛 [Open an Issue](https://github.com/ezzaamizilla/football-player-value-predictor/issues)
- 💬 [Join Discussions](https://github.com/ezzaamizilla/football-player-value-predictor/discussions)
- 📧 Email: zakariaezzaami337@gmail.com

### FAQ
**Q: How often are the models updated?**  
A: Currently, models are trained on 2022-2024 data. We plan to update quarterly.

**Q: Can I use this for Fantasy Football?**  
A: While interesting, this predicts real-world market value, not fantasy points.

**Q: What leagues are supported?**  
A: Top 5 European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).

**Q: Is this suitable for professional use?**  
A: This is an educational tool. For professional scouting, combine with expert analysis.

**Q: Can I contribute my own data?**  
A: Yes! Please open an issue to discuss data integration.

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=ezzaamizilla/football-player-value-predictor&type=Date)](https://star-history.com/#ezzaamizilla/football-player-value-predictor&Date)

---

## 📈 Project Stats

![GitHub repo size](https://img.shields.io/github/repo-size/ezzaamizilla/football-player-value-predictor)
![GitHub contributors](https://img.shields.io/github/contributors/ezzaamizilla/football-player-value-predictor)
![GitHub stars](https://img.shields.io/github/stars/ezzaamizilla/football-player-value-predictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/ezzaamizilla/football-player-value-predictor?style=social)
![GitHub issues](https://img.shields.io/github/issues/ezzaamizilla/football-player-value-predictor)

---

**⚡ Quick Start**: `git clone` → `pip install -r requirements.txt` → `streamlit run app.py`

**📝 Disclaimer**: This tool is for educational and entertainment purposes only. Market values are statistical estimates and should not be used for actual transfer negotiations, financial decisions, or contractual agreements. Always consult with professional scouts, agents, and analysts for real-world applications.

---

<div align="center">

**Made with ⚽ and 🤖 by Ezzaami Zakaria**

[⬆ Back to Top](#-football-player-market-value-predictor)

</div>
