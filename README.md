# Emotion-Based Movie Recommender System

A personalized movie recommendation system that integrates real-time facial emotion recognition with content-based filtering. The system analyzes the user's facial expression via a webcam, maps the dominant emotion to movie genres using The Movie Database (TMDB) API, allows the user to pick a movie, and subsequently serves content-based recommendations using TF-IDF vectorization and cosine similarity on plot overviews. User preferences are logged in a MySQL database to personalize future recommendation sessions.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Emotion-to-Genre Mapping](#emotion-to-genre-mapping)
- [Recommendation Methodology](#recommendation-methodology)
- [Database Schema](#database-schema)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Script Comparison](#script-comparison)
- [Security and Best Practices](#security-and-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Traditional recommendation engines rely primarily on static history or explicit ratings. This project introduces a dynamic affective computing layer:
1. Detects the user's emotional state in real time via computer vision.
2. Surfaces popular movies matching the mood from TMDB.
3. Excludes movies the user has already engaged with.
4. Uses Natural Language Processing (NLP) over movie synopses to recommend topically and thematically similar titles based on user selection.
5. Persists user selections and associated emotional states into a MySQL database.

---

## Key Features

- **Real-Time Facial Emotion Detection**: Utilizes the DeepFace library and OpenCV to capture frames via webcam and classify facial expressions.
- **Dynamic TMDB Genre Matching**: Maps detected emotions (Happy, Sad, Angry, Surprise, Fear, Neutral, Disgust) to specific TMDB genre classifications.
- **Content-Based Filtering**: Employs TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and Cosine Similarity on movie overviews to recommend contextually relevant titles.
- **Preference Persistence**: Logs chosen movie IDs, titles, and emotional context to a MySQL database to filter out previously watched or selected movies in future recommendations.
- **Fallback and Verification Mechanisms**: Allows users to confirm or manually correct detected emotions to ensure recommendation accuracy.

---

## System Architecture

```
+------------------+
|  User / Webcam   |
+--------+---------+
         |
         v
+-----------------------------+
| Real-time Emotion Detection |  (OpenCV + DeepFace)
+--------+--------------------+
         | Dominant Emotion
         v
+-----------------------------+
| TMDB Genre Query & Filter   |  (TMDB Discover API)
+--------+--------------------+
         | Filter out previously chosen movies (MySQL)
         v
+-----------------------------+
|  Top 5 Recommendations      |
+--------+--------------------+
         | User Selects Preferred Movie
         v
+-----------------------------+     +-----------------------------+
| Content-Based Filtering     | --> | Database Persistence        |
| (TF-IDF + Cosine Similarity)|     | (MySQL `preferences` table) |
+-----------------------------+     +-----------------------------+
```

---

## Emotion-to-Genre Mapping

The system associates detected emotional states with corresponding movie genres via TMDB genre IDs:

| Emotion   | TMDB Genre ID | Genre Name | Rationale / Vibe                 |
|:----------|:--------------|:-----------|:---------------------------------|
| Happy     | 35            | Comedy     | Uplifting, lighthearted content  |
| Sad       | 18            | Drama      | Emotional, narrative-driven      |
| Angry     | 28            | Action     | High-energy, tension-releasing   |
| Surprise  | 10749         | Romance    | Engaging, dynamic narratives     |
| Fear      | 27            | Horror     | Thrilling, suspenseful themes    |
| Neutral   | 9648          | Mystery    | Engaging puzzles and plots       |
| Disgust   | 53            | Thriller   | Edgy, gripping storylines        |

*Note: If an unrecognized emotion is encountered, the system defaults to Drama (ID 18).*

---

## Recommendation Methodology

The recommendation engine works in two phases:

1. **Emotion-Driven Discovery**:
   - Queries `/discover/movie` from TMDB using the genre mapped from the user's emotion.
   - Results are sorted by popularity descending.
   - Cross-references against the local MySQL `preferences` table to remove movies the user has already chosen.
   - Returns the top 5 candidates.

2. **Content-Based Similarity**:
   - When the user selects a preferred movie from the initial recommendations, the engine collects the plot overviews (`overview` field) of candidate movies.
   - `TfidfVectorizer(stop_words="english")` converts the text into numerical feature vectors.
   - `cosine_similarity()` calculates pairwise similarity scores between the selected movie vector and all candidate vectors.
   - The top 5 highest-scoring titles (excluding the selected movie itself) are presented as similar recommendations.

---

## Database Schema

The application uses MySQL to store user choices.

### Database Name
`movie_recommender`

### Table: `preferences`

```sql
CREATE DATABASE IF NOT EXISTS movie_recommender;
USE movie_recommender;

CREATE TABLE IF NOT EXISTS preferences (
    movie_id INT PRIMARY KEY,
    movie_name VARCHAR(255) NOT NULL,
    emotion VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Project Structure

```
EmotionBasedRecommender/
│
├── code001.py          # Initial implementation with continuous webcam loop
├── improvedcode.py     # Refined implementation with timed sampling & confirmation
└── README.md           # Project documentation
```

---

## Prerequisites

- **Operating System**: Windows, Linux, or macOS
- **Python**: Version 3.8 to 3.11 recommended
- **Webcam**: Functional camera for real-time facial capture
- **MySQL Server**: Local or remote MySQL instance running
- **TMDB API Key**: Free API key from [The Movie Database](https://www.themoviedb.org/documentation/api)

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/LakshayChakravarti/EmotionBasedRecommender.git
cd EmotionBasedRecommender
```

### 2. Create and Activate a Virtual Environment

On Windows (Command Prompt / PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install opencv-python deepface requests mysql-connector-python scikit-learn pandas tf-keras tensorflow
```

*Note: DeepFace relies on a TensorFlow backend. On initial run, DeepFace will automatically download pre-trained model weights (VGG-Face / Emotion weights).*

### 4. Set Up the MySQL Database

Open your MySQL client and execute the following SQL commands:

```sql
CREATE DATABASE movie_recommender;
USE movie_recommender;

CREATE TABLE preferences (
    movie_id INT PRIMARY KEY,
    movie_name VARCHAR(255) NOT NULL,
    emotion VARCHAR(50) NOT NULL
);
```

---

## Configuration

Both `code001.py` and `improvedcode.py` contain configuration variables near the top of the files:

```python
# TMDB API Configuration
TMDB_API_KEY = "YOUR_TMDB_API_KEY"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# MySQL Database Configuration
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_MYSQL_PASSWORD",
    "database": "movie_recommender",
}
```

Update `MYSQL_CONFIG` with your local MySQL credentials and insert your personal `TMDB_API_KEY`.

---

## Usage

You can run either implementation based on your workflow preference:

### Running the Refined Version (`improvedcode.py`)

This version samples your webcam for 5 seconds, computes the dominant emotion using statistical mode, and asks for confirmation before fetching recommendations.

```bash
python improvedcode.py
```

**Step-by-step Execution**:
1. The camera window opens for 5 seconds. Look toward the camera.
2. The system calculates your dominant emotion.
3. In the console, confirm the detected emotion (`y/n`). If incorrect, select an emotion from the numbered menu.
4. View the top 5 recommended movies based on your emotion.
5. Enter the number corresponding to your chosen movie (1-5).
6. Your choice is saved to MySQL, and content-based recommendations (similar movies) are immediately computed and displayed.

### Running the Continuous Loop Version (`code001.py`)

This version keeps the camera feed open indefinitely until you manually press `q`.

```bash
python code001.py
```

**Step-by-step Execution**:
1. The camera feed launches with an emotion overlay.
2. Position your face in the frame, then press `q` to capture and proceed.
3. View your past preference history (loaded from MySQL).
4. Review the top 5 recommendations along with synopsis summaries.
5. Enter the index of your preferred movie.
6. The system records your choice and outputs similar titles based on plot similarity.

---

## Script Comparison

| Feature / Behavior | `code001.py` | `improvedcode.py` |
|:---|:---|:---|
| **Camera Trigger** | Continuous video feed; stops when pressing `q` | 5-second automatic capture window |
| **Emotion Aggregation** | Dominant emotion from final processed frame | Statistical mode across all frames captured in window |
| **Emotion Confirmation** | Automatic (no manual override) | Interactive confirmation (`y/n`) with manual selection fallback |
| **Supported Emotions** | 6 emotions (`happy`, `sad`, `angry`, `surprise`, `fear`, `neutral`) | 7 emotions (adds `disgust` mapped to Thriller) |
| **SQL Insert Strategy** | `INSERT INTO preferences` | `INSERT ... ON DUPLICATE KEY UPDATE` |
| **History Display** | Prints past saved preferences on launch | Keeps focus on current session flow |
| **Recommendation Display** | Title, release year, and full overview | Clean title and release year listing |

---

## Security and Best Practices

1. **Environment Variables**:
   Avoid committing hardcoded database credentials or API keys directly to source control. It is recommended to use `python-dotenv`:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   TMDB_API_KEY = os.getenv("TMDB_API_KEY")
   ```
2. **Database Hardening**:
   Ensure your MySQL user has only the necessary read/write privileges on the `movie_recommender` database rather than using root privileges.

---

## Troubleshooting

- **Camera Not Opening (`Could not open camera`)**:
  - Verify that no other application (e.g., Zoom, Teams, browser) is currently accessing the webcam.
  - If using an external webcam or virtual camera, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.
  - On Windows, check `Settings > Privacy & Security > Camera` to confirm Python has permission to access the camera.

- **MySQL Connection Errors (`Error connecting to MySQL`)**:
  - Ensure the MySQL service is actively running:
    - Windows: Check Services (`services.msc`) or run `net start MySQL80`.
    - Linux: Run `sudo systemctl status mysql`.
  - Double-check port, username, password, and database existence.

- **DeepFace First-Run Latency**:
  - On the first run, DeepFace will download the pre-trained weights for the emotion detection model (~15-30 MB). Ensure an active internet connection.

- **Missing Overviews for Similarity**:
  - If a movie lacks an overview on TMDB, the script safely fills missing values with empty strings (`.fillna("")`) to avoid vectorization errors.
