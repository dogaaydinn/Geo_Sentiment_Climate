# Quick Start Guide - Geo_Sentiment_Climate

Get the project running in **under 10 minutes**! 🚀

---

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git
- 4GB RAM minimum

---

## Option 1: Docker (Recommended) ⚡

**Fastest way to get started!**

```bash
# 1. Clone the repository
git clone https://github.com/dogaaydinn/Geo_Sentiment_Climate.git
cd Geo_Sentiment_Climate

# 2. Start all services
docker-compose up -d

# 3. Check if services are running
docker-compose ps

# 4. Access the API
curl http://localhost:8000/health

# 5. View API documentation
open http://localhost:8000/docs  # or visit in browser
```

**Services Available:**
- 🌐 API: http://localhost:8000
- 📊 MLflow: http://localhost:5000
- 📈 Prometheus: http://localhost:9090
- 📊 Grafana: http://localhost:3000 (admin/admin_change_me)
- 🗄️ PostgreSQL: localhost:5432
- 🔴 Redis: localhost:6379

---

## Option 2: Local Development 💻

**For active development:**

```bash
# 1. Clone and navigate
git clone https://github.com/dogaaydinn/Geo_Sentiment_Climate.git
cd Geo_Sentiment_Climate

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install in development mode
pip install -e .

# 5. Start the API server
python -m uvicorn source.api.main:app --reload --host 0.0.0.0 --port 8000

# 6. In another terminal, verify it's working
curl http://localhost:8000/health
```

---

## Quick Tests ✅

### Test 1: Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","timestamp":"...","version":"2.0.0"}
```

### Test 2: List Models
```bash
curl http://localhost:8000/models
# Expected: [] (empty array if no models registered yet)
```

### Test 3: API Documentation
Visit: http://localhost:8000/docs

You should see interactive Swagger UI with all endpoints.

---

## Train Your First Model 🤖

```bash
# 1. Make sure you have some data in data/processed/
# (or use sample data if available)

# 2. Run training
python -m source.ml.model_training

# 3. Check MLflow for experiment tracking
open http://localhost:5000
```

---

## Make Your First Prediction 🎯

```python
import requests

# Prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "data": {
            "pm25": 35.5,
            "temperature": 25.0,
            "humidity": 60.0,
            "wind_speed": 5.0
        }
    }
)

print(response.json())
# Expected: {"predictions": [...], "model_id": "...", ...}
```

Or use curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "pm25": 35.5,
      "temperature": 25.0,
      "humidity": 60.0
    }
  }'
```

---

## Run Tests 🧪

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=source --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## Development Workflow 🔄

### 1. Create a Feature Branch
```bash
git checkout -b feature/my-awesome-feature
```

### 2. Make Changes
Edit files in `source/` directory

### 3. Run Tests
```bash
pytest tests/
```

### 4. Format Code
```bash
black source/ tests/
flake8 source/ tests/
```

### 5. Commit Changes
```bash
git add .
git commit -m "feat: add awesome feature"
```

### 6. Push and Create PR
```bash
git push origin feature/my-awesome-feature
# Then create PR on GitHub
```

---

## Common Tasks 📋

### Add a New ML Model

1. Create model file:
```python
# source/ml/models/my_model.py
class MyModel:
    def __init__(self, **params):
        self.params = params

    def train(self, X, y):
        # Training logic
        pass

    def predict(self, X):
        # Prediction logic
        pass
```

2. Register in model training:
```python
# source/ml/model_training.py
# Add to create_model() function
elif model_type == "my_model":
    model = MyModel(**params)
```

### Add a New API Endpoint

```python
# source/api/main.py
@app.post("/my-endpoint")
async def my_endpoint(request: MyRequest):
    """My new endpoint."""
    return {"result": "success"}
```

### Add a New Data Source

```python
# source/data_ingestion/my_data_source.py
def ingest_my_data(source_path: Path):
    """Ingest data from my custom source."""
    # Ingestion logic
    pass
```

---

## Troubleshooting 🔧

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Docker Issues
```bash
# Stop all containers
docker-compose down

# Remove volumes (fresh start)
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Start again
docker-compose up -d
```

### Import Errors
```bash
# Reinstall in development mode
pip install -e .

# Or add source to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

---

## Environment Variables 🔐

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```bash
# Minimal required settings
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=postgresql://geo_climate:password@localhost:5432/geo_climate_db
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Next Steps 🎯

After getting the project running:

1. ✅ **Read the Documentation**
   - `README.md` - Project overview
   - `ROADMAP.md` - Long-term strategy
   - `PROJECT_REVIEW.md` - Detailed improvement plan
   - `CONTRIBUTING.md` - How to contribute

2. ✅ **Explore the API**
   - Interactive docs: http://localhost:8000/docs
   - Try different endpoints
   - Understand request/response formats

3. ✅ **Train a Model**
   - Prepare your data
   - Configure training parameters
   - Monitor in MLflow

4. ✅ **Run the Tests**
   - Understand test structure
   - Add tests for new features
   - Maintain >80% coverage

5. ✅ **Set Up Your IDE**
   - Configure Python interpreter
   - Set up debugging
   - Install extensions (Python, Docker, etc.)

---

## Useful Commands 📝

```bash
# Project structure
tree -L 2 -I '__pycache__|*.pyc|.git'

# Count lines of code
find source/ -name "*.py" -exec wc -l {} + | tail -1

# Find TODO comments
grep -r "TODO\|FIXME" source/

# Check code style
black --check source/
flake8 source/

# Type checking
mypy source/

# Security check
bandit -r source/

# Generate requirements
pip freeze > requirements-freeze.txt

# Update dependencies
pip install --upgrade -r requirements.txt

# Clean cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

---

## IDE Setup Recommendations 💡

### VSCode
Install extensions:
- Python
- Pylance
- Docker
- GitLens
- YAML
- Jupyter

Settings (`.vscode/settings.json`):
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true
}
```

### PyCharm
1. Mark `source/` as Sources Root
2. Enable pytest as test runner
3. Configure Black as formatter
4. Enable type checking

---

## Performance Tips ⚡

### Speed Up Docker Builds
```dockerfile
# Use BuildKit
export DOCKER_BUILDKIT=1
docker-compose build
```

### Speed Up pip Install
```bash
# Use pip cache
pip install --cache-dir ~/.cache/pip -r requirements.txt

# Install minimal deps
pip install -r requirements-minimal.txt
```

### Speed Up Tests
```bash
# Run tests in parallel
pytest -n auto

# Run only changed tests
pytest --lf  # last failed
pytest --ff  # failed first
```

---

## Getting Help 💬

### Documentation
- 📖 Project docs in `docs/` folder
- 🌐 API docs at http://localhost:8000/docs
- 📝 README.md and other markdown files

### Community
- 🐛 **Issues**: https://github.com/dogaaydinn/Geo_Sentiment_Climate/issues
- 💬 **Discussions**: GitHub Discussions (if enabled)
- 📧 **Email**: dogaa882@gmail.com

### Debug Mode
```bash
# Start API with debug logging
LOG_LEVEL=DEBUG python -m uvicorn source.api.main:app --reload

# Enable verbose test output
pytest -vv

# Python debugger
python -m pdb source/ml/model_training.py
```

---

## What's Next? 🚀

Choose your path:

### 🎓 **Learn the Codebase**
1. Read `PROJECT_REVIEW.md` for architecture overview
2. Explore `source/` directory structure
3. Review existing tests in `tests/`
4. Check out notebooks in `notebooks/`

### 🛠️ **Start Contributing**
1. Pick an issue from GitHub
2. Read `CONTRIBUTING.md`
3. Create a feature branch
4. Submit a PR

### 🧪 **Improve Testing**
1. Run pytest with coverage
2. Identify untested code
3. Add integration tests
4. Add e2e tests

### 📊 **Work on ML**
1. Train different models
2. Optimize hyperparameters
3. Evaluate model performance
4. Deploy to production

### 🌐 **Enhance the API**
1. Add authentication
2. Implement rate limiting
3. Add new endpoints
4. Improve documentation

---

## Success! 🎉

If you made it here, you should have:
- ✅ Running API at http://localhost:8000
- ✅ MLflow at http://localhost:5000
- ✅ All services in Docker
- ✅ Tests passing
- ✅ Understanding of project structure

**You're ready to start developing!** 🚀

For detailed next steps, see `PROJECT_REVIEW.md` → "Recommended Next Steps".

---

**Happy Coding!** 💻✨

*Last Updated: November 2024*
