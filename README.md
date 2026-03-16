# 🥗 Nutrition Recommendation System

[![Tests](https://github.com/yourusername/nutrition-recommender/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/yourusername/nutrition-recommender/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)

An intelligent, hybrid machine learning system for personalized nutrition recommendations targeting obesity management. Combines ML classification, collaborative filtering, and content-based methods to provide tailored dietary guidance.

## 🎯 Features

- **ML Classification**: Gradient Boosting classifier (95%+ accuracy) predicts obesity levels
- **Collaborative Filtering**: KNN-based discovery of similar user profiles
- **Content-Based Recommendations**: Rule-based nutrition plans for each obesity category
- **Personalized Alerts**: Habit-based notifications (low hydration, high junk food, etc.)
- **Web Dashboard**: Interactive FastAPI + HTML/CSS interface with real-time BMI calculations
- **Docker Ready**: Production-ready containerization with CI/CD pipeline
- **Power BI/Tableau Integration**: REST API endpoints for enterprise BI tools

## 📊 Dataset

**UCI Machine Learning Repository** - "Estimation of Obesity Levels Based on Eating Habits and Physical Condition"
- **Records**: 2,111 individuals
- **Features**: 17 (demographics + behavioral + calculated)
- **Target**: 7 obesity levels (Insufficient Weight → Obesity Type III)
- **Source**: https://archive.ics.uci.edu/dataset/544

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (optional)
- Git

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nutrition-recommender.git
   cd nutrition-recommender
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python train_and_save_model.py
   ```

5. **Run tests**
   ```bash
   pytest tests/ -v
   ```

6. **Start FastAPI server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

Visit `http://localhost:8000` in your browser.

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t nutrition-recommender:latest .
   ```

2. **Run the container**
   ```bash
   docker run -d -p 8000:8000 --name nutrition-api nutrition-recommender:latest
   ```

3. **Using Docker Compose** (for dev environment)
   ```bash
   docker-compose up -d
   ```

### Access the Application

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Web Dashboard**: http://localhost:8000 (HTML interface)

## 📁 Project Structure

```
nutrition-recommender/
├── .github/workflows/           # GitHub Actions CI/CD
│   └── ci-cd.yml               # Main pipeline
├── app/                         # Web application
│   ├── main.py                 # FastAPI server
│   └── dashboard.py            # Streamlit dashboard
├── src/                         # Core modules
│   ├── model.py                # ML models
│   ├── train.py                # Training pipeline
│   ├── preprocess.py           # Data preprocessing
│   └── eda_summary.py          # Analysis
├── tests/                       # Test suite
│   ├── test_api.py             # API tests
│   └── __init__.py
├── models/                      # Trained artifacts
│   ├── nutrition_model.pkl     # Serialized models
│   └── cosine_sim.npy          # Similarity matrix
├── Data/                        # Raw & processed data
│   ├── ObesityDataSet*.csv     # UCI dataset
│   └── processed/              # Preprocessed data
├── notebooks/                   # Jupyter notebooks
│   └── nutrition_recommendation_model.ipynb
├── reports/                     # Metrics & reports
│   ├── metrics.csv
│   ├── test_predictions.csv
│   └── training_history.csv
├── mlruns/                      # MLflow experiment tracking
├── Dockerfile                   # Container definition
├── docker-compose.yml          # Multi-container setup
├── requirements.txt            # Python dependencies
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
├── main.py                     # Entry point
├── train_and_save_model.py     # Model training script
├── test_model_validation.py    # Validation tests
└── README.md                   # This file
```

## 🔧 API Endpoints

### Health & Status
```bash
GET /health
# Returns: {"status": "healthy", "model_loaded": true}
```

### BMI Calculator
```bash
GET /bmi?height_cm=180&weight_kg=75
# Returns: {"bmi": 23.15, "category": "Normal Weight"}
```

### Get Recommendation
```bash
POST /recommend
Content-Type: application/json

{
  "gender": "Male",
  "age": 35,
  "height": 1.70,
  "weight": 110,
  "family_history": "yes",
  "favc": "yes",
  "fcvc": 1,
  "ncp": 3,
  "caec": "Always",
  "smoke": "no",
  "ch2o": 1,
  "scc": "no",
  "faf": 0,
  "tue": 3,
  "calc": "Frequently",
  "mtrans": "Automobile"
}

# Returns personalized nutrition plan + similar users
```

### Find Similar Users
```bash
GET /similar-users?user_index=42&k=5
# Returns k most similar user profiles
```

## 📈 Model Performance

| Metric | Gradient Boosting | Random Forest | MLP |
|--------|-------------------|---------------|-----|
| Test Accuracy | 94.78% | 93.21% | 91.56% |
| Macro F1 | 94.32% | 92.65% | 90.87% |
| Weighted F1 | 94.81% | 93.34% | 91.72% |

**Top Features**:
1. BMI (23.4%) 2. Weight (18.7%) 3. Age (15.6%) 4. Height (14.2%) 5. Physical Activity (9.8%)

## 🔄 CI/CD Pipeline

Automated testing, building, and deployment via GitHub Actions:

1. **Test Stage**: Python tests + model validation
2. **Build Stage**: Docker image creation
3. **Push Stage**: Docker Hub registry push
4. **Deploy Stage**: Server deployment (customizable)

Triggered automatically on:
- `push` to `main` or `master`
- `pull_request` to main/master

### Required GitHub Secrets:
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token (recommended) or password

## 📊 BI Integration

### Power BI
- **Data Connector**: REST API → Power Query
- **Custom Visuals**: Deneb charts for BMI distribution
- **DAX Measures**: Recommendation adherence scoring
- **Power Automate**: Automated report generation

### Tableau
- **Web Data Connector**: FastAPI integration
- **Dashboard Actions**: Interactive user filtering
- **Story Points**: User journey narratives
- **Parameter Controls**: Threshold adjustments

Connect to: `http://api-server:8000/predict` endpoint

## 🧪 Testing

Run the full test suite:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

Test specific module:
```bash
pytest tests/test_api.py::test_health_check_returns_valid_structure -v
```

## 📝 Model Training

Full pipeline with MLflow tracking:
```bash
python train_and_save_model.py --epochs 100 --batch-size 32 --log-mlflow
```

View experiments:
```bash
mlflow ui
```

## 🐳 Docker Build Arguments

Optimize build size and speed:
```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg SKLEARN_VERSION=1.4.2 \
  -t nutrition-recommender:latest .
```

## 📦 Production Deployment

### AWS EC2
```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.small \
  --key-name your-key
```

### Docker Swarm
```bash
docker swarm init
docker service create \
  -p 8000:8000 \
  --name nutrition-api \
  nutrition-recommender:latest
```

### Kubernetes
```bash
kubectl apply -f k8s-deployment.yaml
kubectl expose deployment nutrition-api --type=LoadBalancer --port=8000
```

## 🔐 Security Best Practices

- ✅ Load secrets from environment variables (`.env`)
- ✅ Use HTTPS in production
- ✅ Validate all user inputs
- ✅ Rate limiting on API endpoints
- ✅ CORS configuration for web dashboards
- ✅ Non-root container user

## 📚 Documentation

- [API Documentation](./docs/API.md) - Detailed endpoint specs
- [Architecture](./docs/ARCHITECTURE.md) - System design
- [Development Guide](./CONTRIBUTING.md) - Contributing guidelines
- [Deployment Guide](./docs/DEPLOYMENT.md) - Production setup

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on:
- Code style (PEP 8)
- Pull request process
- Issue reporting
- Feature requests

## 📄 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file.

## 📧 Contact & Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the obesity dataset
- scikit-learn, pandas, FastAPI communities
- Contributors and testers

---

**⭐ If you find this project useful, please consider starring the repository!**

