{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f4b02de7-9e4c-4b90-9274-d1e9c82fbd15",
   "metadata": {},
   "source": [
    "#### Business Understanding\n",
    "#### Project Title\n",
    "\n",
    "##### Blood Donation Prediction\n",
    "\n",
    "#### Problem Statement\n",
    "\n",
    "Blood donation is essential for saving lives. Blood banks need to maintain an adequate supply of blood for emergencies, surgeries, and treatments.\n",
    "\n",
    "The objective of this project is to predict whether a donor will donate blood again in March 2007 based on their donation history.\n",
    "\n",
    "#### Business Objective\n",
    "\n",
    "The Blood Transfusion Service Center wants to identify donors who are likely to donate blood again.\n",
    "\n",
    "This helps:\n",
    "\n",
    "- Improve blood collection campaigns\n",
    "- Target potential donors\n",
    "- Reduce marketing costs\n",
    "- Maintain sufficient blood supply\n",
    "\n",
    "#### Machine Learning Objective\n",
    "\n",
    "Build a classification model that predicts:\n",
    "\n",
    "Target Variable\n",
    "\n",
    "Made Donation in March 2007\n",
    "\n",
    "Value\tMeaning\n",
    "0\tDid Not Donate\n",
    "1\tDonated\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c14e56e5-51e8-449c-aaa7-55d805771db7",
   "metadata": {},
   "source": [
    "#### Import Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f491b66c-0df5-4ef3-be44-75a11d681c34",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3f228ff-5d01-4e41-8443-e2e8b519e3dd",
   "metadata": {},
   "source": [
    "#### Load Dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c89c2a1b-3007-405e-81ca-ae1a86c391c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"Warm_Up_Predict_Blood_Donations_-_Traning_Data.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1fcc0ea0-7dde-4870-935b-27cf02394351",
   "metadata": {},
   "source": [
    "#### View Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "63010523-f39a-4397-a030-04031293fa09",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Months since Last Donation</th>\n",
       "      <th>Number of Donations</th>\n",
       "      <th>Total Volume Donated (c.c.)</th>\n",
       "      <th>Months since First Donation</th>\n",
       "      <th>Made Donation in March 2007</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>619</td>\n",
       "      <td>2</td>\n",
       "      <td>50</td>\n",
       "      <td>12500</td>\n",
       "      <td>98</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>664</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>3250</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>441</td>\n",
       "      <td>1</td>\n",
       "      <td>16</td>\n",
       "      <td>4000</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>160</td>\n",
       "      <td>2</td>\n",
       "      <td>20</td>\n",
       "      <td>5000</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>358</td>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>6000</td>\n",
       "      <td>77</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Months since Last Donation  Number of Donations  \\\n",
       "0         619                           2                   50   \n",
       "1         664                           0                   13   \n",
       "2         441                           1                   16   \n",
       "3         160                           2                   20   \n",
       "4         358                           1                   24   \n",
       "\n",
       "   Total Volume Donated (c.c.)  Months since First Donation  \\\n",
       "0                        12500                           98   \n",
       "1                         3250                           28   \n",
       "2                         4000                           35   \n",
       "3                         5000                           45   \n",
       "4                         6000                           77   \n",
       "\n",
       "   Made Donation in March 2007  \n",
       "0                            1  \n",
       "1                            1  \n",
       "2                            1  \n",
       "3                            1  \n",
       "4                            0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d6870650-f090-470c-b9ff-3c8c7253f45f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Months since Last Donation</th>\n",
       "      <th>Number of Donations</th>\n",
       "      <th>Total Volume Donated (c.c.)</th>\n",
       "      <th>Months since First Donation</th>\n",
       "      <th>Made Donation in March 2007</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>571</th>\n",
       "      <td>698</td>\n",
       "      <td>23</td>\n",
       "      <td>1</td>\n",
       "      <td>250</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>572</th>\n",
       "      <td>433</td>\n",
       "      <td>16</td>\n",
       "      <td>3</td>\n",
       "      <td>750</td>\n",
       "      <td>86</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>573</th>\n",
       "      <td>360</td>\n",
       "      <td>21</td>\n",
       "      <td>2</td>\n",
       "      <td>500</td>\n",
       "      <td>52</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>574</th>\n",
       "      <td>541</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>250</td>\n",
       "      <td>39</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>575</th>\n",
       "      <td>74</td>\n",
       "      <td>72</td>\n",
       "      <td>1</td>\n",
       "      <td>250</td>\n",
       "      <td>72</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Unnamed: 0  Months since Last Donation  Number of Donations  \\\n",
       "571         698                          23                    1   \n",
       "572         433                          16                    3   \n",
       "573         360                          21                    2   \n",
       "574         541                          39                    1   \n",
       "575          74                          72                    1   \n",
       "\n",
       "     Total Volume Donated (c.c.)  Months since First Donation  \\\n",
       "571                          250                           23   \n",
       "572                          750                           86   \n",
       "573                          500                           52   \n",
       "574                          250                           39   \n",
       "575                          250                           72   \n",
       "\n",
       "     Made Donation in March 2007  \n",
       "571                            0  \n",
       "572                            0  \n",
       "573                            0  \n",
       "574                            0  \n",
       "575                            0  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b69c735-0272-423d-ad35-1d5ece41373d",
   "metadata": {},
   "source": [
    "The dataset contains donor information such as donation frequency, recency, total volume donated, and donation history.\n",
    "\n",
    "The target variable indicates whether the donor donated blood in March 2007."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "667c371c-14a6-410d-b6ba-95d10afc3c54",
   "metadata": {},
   "source": [
    "#### Data Shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1f397419-86f2-46df-98b4-7999eb83bc22",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(576, 6)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a027e0a-c586-416b-a7c4-5a8d68ce4e4e",
   "metadata": {},
   "source": [
    "- The dataset contains multiple donor records and several attributes related to donation history."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "898ce9c2-fcb3-4c2b-9e40-6aff6c6d47a7",
   "metadata": {},
   "source": [
    "#### Dataset Information"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "edcee827-28d5-4d40-a8bf-921e85dc5c3e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 576 entries, 0 to 575\n",
      "Data columns (total 6 columns):\n",
      " #   Column                       Non-Null Count  Dtype\n",
      "---  ------                       --------------  -----\n",
      " 0   Unnamed: 0                   576 non-null    int64\n",
      " 1   Months since Last Donation   576 non-null    int64\n",
      " 2   Number of Donations          576 non-null    int64\n",
      " 3   Total Volume Donated (c.c.)  576 non-null    int64\n",
      " 4   Months since First Donation  576 non-null    int64\n",
      " 5   Made Donation in March 2007  576 non-null    int64\n",
      "dtypes: int64(6)\n",
      "memory usage: 27.1 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "468918a5-d081-4e39-9dd9-aea59ecc112d",
   "metadata": {},
   "source": [
    "- All features are numerical.\n",
    "\n",
    "- No datatype conversion is required."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1226d15c-0f03-47d0-aa60-fd4729d35dab",
   "metadata": {},
   "source": [
    "#### Statistical Summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f236e2d2-612a-4d67-b341-e564e5263eba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Months since Last Donation</th>\n",
       "      <th>Number of Donations</th>\n",
       "      <th>Total Volume Donated (c.c.)</th>\n",
       "      <th>Months since First Donation</th>\n",
       "      <th>Made Donation in March 2007</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>576.000000</td>\n",
       "      <td>576.000000</td>\n",
       "      <td>576.000000</td>\n",
       "      <td>576.000000</td>\n",
       "      <td>576.000000</td>\n",
       "      <td>576.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>374.034722</td>\n",
       "      <td>9.439236</td>\n",
       "      <td>5.427083</td>\n",
       "      <td>1356.770833</td>\n",
       "      <td>34.050347</td>\n",
       "      <td>0.239583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>216.947773</td>\n",
       "      <td>8.175454</td>\n",
       "      <td>5.740010</td>\n",
       "      <td>1435.002556</td>\n",
       "      <td>24.227672</td>\n",
       "      <td>0.427200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>250.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>183.750000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>500.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>375.500000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>562.500000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1750.000000</td>\n",
       "      <td>49.250000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>747.000000</td>\n",
       "      <td>74.000000</td>\n",
       "      <td>50.000000</td>\n",
       "      <td>12500.000000</td>\n",
       "      <td>98.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Unnamed: 0  Months since Last Donation  Number of Donations  \\\n",
       "count  576.000000                  576.000000           576.000000   \n",
       "mean   374.034722                    9.439236             5.427083   \n",
       "std    216.947773                    8.175454             5.740010   \n",
       "min      0.000000                    0.000000             1.000000   \n",
       "25%    183.750000                    2.000000             2.000000   \n",
       "50%    375.500000                    7.000000             4.000000   \n",
       "75%    562.500000                   14.000000             7.000000   \n",
       "max    747.000000                   74.000000            50.000000   \n",
       "\n",
       "       Total Volume Donated (c.c.)  Months since First Donation  \\\n",
       "count                   576.000000                   576.000000   \n",
       "mean                   1356.770833                    34.050347   \n",
       "std                    1435.002556                    24.227672   \n",
       "min                     250.000000                     2.000000   \n",
       "25%                     500.000000                    16.000000   \n",
       "50%                    1000.000000                    28.000000   \n",
       "75%                    1750.000000                    49.250000   \n",
       "max                   12500.000000                    98.000000   \n",
       "\n",
       "       Made Donation in March 2007  \n",
       "count                   576.000000  \n",
       "mean                      0.239583  \n",
       "std                       0.427200  \n",
       "min                       0.000000  \n",
       "25%                       0.000000  \n",
       "50%                       0.000000  \n",
       "75%                       0.000000  \n",
       "max                       1.000000  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac8ec9ac-a673-4734-816e-beeb251e347c",
   "metadata": {},
   "source": [
    "- The summary statistics provide information about donor behavior and donation frequency."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "03cce9c6-ee4e-4221-99ca-57230ee27d56",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Unnamed: 0                     0\n",
       "Months since Last Donation     0\n",
       "Number of Donations            0\n",
       "Total Volume Donated (c.c.)    0\n",
       "Months since First Donation    0\n",
       "Made Donation in March 2007    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e6b091e5-f877-4d58-974f-e824061af2c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "438e5535-7ee9-4695-9b19-e96b43286756",
   "metadata": {},
   "source": [
    "#### Removing Unneccessary Columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f72b83de-997c-4a6a-bd1f-a46185f2270a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop('Unnamed: 0',axis=1,inplace =True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab5589a7-40c6-4db1-b7c7-eb6149e8a41d",
   "metadata": {},
   "source": [
    "### Exploratory Data Analysis(EDA)\n",
    "#### Understanding the Target Variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "14239efd-97ec-4a24-95c4-8f1faf94b041",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Made Donation in March 2007\n",
       "0    438\n",
       "1    138\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Made Donation in March 2007'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "4c14079d-f347-4926-be77-4d7e800c0dee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAHFCAYAAAAUpjivAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAw/ElEQVR4nO3dfXzPdf////vb2AzbO9ts885yEok21HRxULIak5xVOhCJg5ImtcPp16ETjj44KCd1KEcUElp9itKZC3KS09JYIenkMyFbU2ZzMtvM8/dHv72O3jZhxnuebtfL5X25eD1fj/fr9Xi93q33fa+zuYwxRgAAAJaq4OsGAAAALibCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOUA7MmzdPLpfLeVWuXFmRkZG6/fbbNXHiRGVmZvq6RS+LFi3S9OnTS5zncrk0duzYS9qPVLp9OHbsWLlcrvNaz/HjxzV27FitWbPmvN5X0rrq1KmjTp06nddyzqY8fjaAr1X0dQMA/mvu3Lm6/vrrVVBQoMzMTK1fv16TJk3S888/r7feektt27b1dYuSfv9C3bFjh5KSkorN27Rpk2rVqnXpm/r/nc8+fOihh3TnnXee1/KPHz+ucePGSZLi4uLO+X2lWVdplOfPBvAVwg5QjkRHR6t58+bOdLdu3fT3v/9dt956q+699159//33ioiI8GGHZ/eXv/zFp+s/n31Yq1ati/7lf/z4cVWpUuWSrOtsfP3ZAL7CaSygnLvmmms0ZcoUHTlyRK+88orXvKVLl6ply5aqUqWKgoKC1K5dO23atMmrpuj0yc6dO3X//ffL7XYrIiJC/fv3V3Z2tlftSy+9pNtuu03h4eGqWrWqYmJiNHnyZBUUFDg1cXFx+uijj/TTTz95nTYqUtKpkh07dqhr166qXr26KleurGbNmun111/3qlmzZo1cLpfefPNNjRkzRh6PR8HBwWrbtq127959IbvwjPuwpFNLq1atUlxcnEJDQxUYGKhrrrlG3bp10/Hjx7Vnzx7VqFFDkjRu3Dhn2/v16+e1vK1bt+q+++5T9erVde21155xXUWWLFmiJk2aqHLlyqpXr55efPFFr/lFp+j27NnjNV60z4pOqV2Onw1wKXBkB7gM3HXXXfLz89Nnn33mjC1atEi9e/dWQkKC3nzzTeXl5Wny5MmKi4vTp59+qltvvdVrGd26dVOPHj00YMAAbd++XaNHj5YkzZkzx6n58ccf1atXL9WtW1f+/v766quvNH78eH377bdO3csvv6yBAwfqxx9/1JIlS87a++7du9WqVSuFh4frxRdfVGhoqBYsWKB+/frpl19+0ciRI73q//GPf+iWW27Rq6++qpycHI0aNUqdO3fWrl275OfnV6b78HR79uxRx44d1bp1a82ZM0dXXXWVfv75Zy1btkz5+fmqWbOmli1bpjvvvFMDBgzQQw89JElOACpy7733qmfPnho0aJCOHTv2p32lpqYqKSlJY8eOVWRkpBYuXKgnnnhC+fn5Gj58+Hlt4+X62QAXnQHgc3PnzjWSzJYtW85YExERYRo1amSMMaawsNB4PB4TExNjCgsLnZojR46Y8PBw06pVK2fsmWeeMZLM5MmTvZaXmJhoKleubE6dOlXi+goLC01BQYGZP3++8fPzM4cOHXLmdezY0dSuXbvE90kyzzzzjDPds2dPExAQYPbu3etV16FDB1OlShVz+PBhY4wxq1evNpLMXXfd5VX39ttvG0lm06ZNZ9gzvzvffWjMf/dNkXfeecdIMqmpqWdcxsGDB4tt4+nLe/rpp884749q165tXC5XsfW1a9fOBAcHm2PHjnltW1pamldd0T5bvXq1M1YePxvA1ziNBVwmjDHOv3fv3q0DBw6oT58+qlDhvz/G1apVU7du3bR582YdP37c6/1dunTxmm7SpIlOnDjhdZfStm3b1KVLF4WGhsrPz0+VKlXSgw8+qMLCQn333Xel6nvVqlWKj49XVFSU13i/fv10/PjxYqfdSupTkn766adSrf+P/rgPS9KsWTP5+/tr4MCBev311/V///d/pVpPt27dzrn2hhtuUNOmTb3GevXqpZycHG3durVU6z9X5emzAS4mwg5wGTh27Jh+++03eTweSdJvv/0mSapZs2axWo/Ho1OnTikrK8trPDQ01Gs6ICBAkpSbmytJ2rt3r1q3bq2ff/5ZL7zwgtatW6ctW7bopZde8qo7X7/99tsZ+/zjtpxrn6V1+j4sybXXXquVK1cqPDxcgwcP1rXXXqtrr71WL7zwwnmtq6TtPZPIyMgzjp2+b8paeflsgIuNa3aAy8BHH32kwsJC51bnoi+d9PT0YrUHDhxQhQoVVL169fNax3vvvadjx45p8eLFql27tjOemppa6r6Lej1Tn5IUFhZ2Qcs/V6fvwzNp3bq1WrdurcLCQn355Zf697//raSkJEVERKhnz57ntK7zeXZPRkbGGceKPufKlStLkvLy8rzqfv3113NeT0nKy2cDXGwc2QHKub1792r48OFyu9165JFHJEkNGzbU1VdfrUWLFnmdmjl27Jjeffdd5w6t81H0BV3027r0+2mf2bNnF6sNCAg459/m4+PjtWrVKucLtMj8+fNVpUqVS3I7dEn78Gz8/PzUokUL58hW0Smlsj6asXPnTn311VdeY4sWLVJQUJBuuukmSb8/fFCSvv76a6+6pUuXFlve5fbZAJcCR3aAcmTHjh06efKkTp48qczMTK1bt05z586Vn5+flixZ4tz1U6FCBU2ePFm9e/dWp06d9MgjjygvL0/PPfecDh8+rH/961/nve527drJ399f999/v0aOHKkTJ05o5syZxU6HSVJMTIwWL16smTNnKjY2VhUqVPB6ts0fPfPMM/rwww91++236+mnn1ZISIgWLlyojz76SJMnT5bb7T7vXv/Mue7DkvznP//RqlWr1LFjR11zzTU6ceKEcxda0cMIg4KCVLt2bb3//vuKj49XSEiIwsLCnEByvjwej7p06aKxY8eqZs2aWrBggVasWKFJkyY5gfXmm29Ww4YNNXz4cJ08eVLVq1fXkiVLtH79+mLLK8+fDeAzvr0+GoAx/73bpujl7+9vwsPDTZs2bcyECRNMZmZmie977733TIsWLUzlypVN1apVTXx8vNmwYYNXTdFdQAcPHixxnX+8w+eDDz4wTZs2NZUrVzZXX321GTFihPnkk0+K3fFz6NAhc99995mrrrrKuFwur7uMVMKdStu3bzedO3c2brfb+Pv7m6ZNm5q5c+d61RTd8fO///u/XuNpaWlGUrH6stiHp98htWnTJnPPPfeY2rVrm4CAABMaGmratGljli5d6vW+lStXmhtvvNEEBAQYSaZv375eyzt9X5e0LmN+vxurY8eO5p133jE33HCD8ff3N3Xq1DFTp04t9v7vvvvOJCQkmODgYFOjRg0zZMgQ89FHH10Wnw3gay5jznJ7AgAAwGWMa3YAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKzGQwUlnTp1SgcOHFBQUNB5PeYdAAD4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p556Slu3btXixYv13XffqUuXLl51SUlJWrJkiZKTk7V+/XodPXpUnTp1UmFhoVPTq1cvpaamatmyZVq2bJlSU1PVp08fZ36PHj2Unp7u9Wrfvr3atGmj8PDwMt82oBgD4IL17dvXdO3a1cTExJgFCxY44wsXLjQxMTGma9eupm/fvs74J598Ym655RbjdrtNSEiI6dixo/nhhx+8lvn555+bZs2amYCAABMbG2sWL15sJJlt27Y5NTt37jQdOnQwVatWNeHh4eaBBx4wBw8ePGOfq1evNpJMVlaW1/jx48dNw4YNzS233OKMFRYWmnHjxpmrr77a+Pv7m6ZNm5pPPvnEmZ+WlmYkmXfffdfExcWZwMBA06RJE7Nx40an5tdffzU9e/Y0V199tQkMDDTR0dFm0aJFXvtNktcrLS2txD7feecd07hxY+Pv729q165tnn/+ea9tqF27thk/frz529/+ZqpVq2aioqLMK6+8csZ9YYwxbdq0MU888cQFLaOsP/ui/frWW2+ZNm3amICAADNnzhxjjDGvvfaasw8iIyPN4MGDnfdJMrNnzzZ33323CQwMNPXr1zfvv//+n/Z+ui+++MJIMj/99JMxxpjDhw+bSpUqmeTkZKfm559/NhUqVDDLli0zxhjzzTffGElm8+bNTs2mTZuMJPPtt9+WuJ7MzExTqVIlM3/+/PPqDygtjuwAZehvf/ub5s6d60zPmTNH/fv3L1Z37NgxDR06VFu2bNGnn36qChUq6J577tGpU6ec+Z06dVLDhg2VkpKisWPHavjw4V7LSE9PV5s2bdSsWTN9+eWXWrZsmX755Rd17979vPsODAzUoEGDtGHDBue0wgsvvKApU6bo+eef19dff6327durS5cu+v77773eO2bMGA0fPlypqam67rrrdP/99+vkyZOSfj+6ERsbqw8//FA7duzQwIED1adPH33++efOOlq2bKmHH37Y+Y0/KiqqWH8pKSnq3r27evbsqe3bt2vs2LF66qmniv2BxylTpqh58+batm2bEhMT9eijj+rbb789r31R2mWU1WdfZNSoUXr88ce1a9cutW/fXjNnztTgwYM1cOBAbd++XUuXLlX9+vW93jNu3Dh1795dX3/9te666y717t1bhw4dOudtz87Olsvl0lVXXSXp9/1eUFCghIQEp8bj8Sg6OlobN26UJG3atElut1stWrRwav7yl7/I7XY7NaebP3++qlSpovvuu++cewMuiK/TFmCDot/uDx48aAICAkxaWprZs2ePqVy5sjl48GCx3+5Pl5mZaSSZ7du3G2OMeeWVV0xISIg5duyYUzNz5kyvIztPPfWUSUhI8FrOvn37jCSze/fuEtdzpiM7xvx+xEGS+fzzz40xxng8HjN+/HivmptvvtkkJiYaY/57BOLVV1915u/cudNIMrt27Trjtt51111m2LBhzvTpR1dK6rNXr16mXbt2XjUjRowwjRs3dqZr165tHnjgAWf61KlTJjw83MycOfOMvZR0ZOd8l1HWn33Rfp0+fbpXncfjMWPGjDnjciSZJ5980pk+evSocblcXkfj/kxubq6JjY01vXv3dsYWLlxo/P39i9W2a9fODBw40BhjzPjx402DBg2K1TRo0MBMmDChxHU1btzYPProo+fUF1AWOLIDlKGwsDB17NhRr7/+uubOnauOHTsqLCysWN2PP/6oXr16qV69egoODlbdunUlSXv37pUk7dq1S02bNlWVKlWc97Rs2dJrGSkpKVq9erWqVavmvK6//npn+efLGCPp97t6cnJydODAAd1yyy1eNbfccot27drlNdakSRPn3zVr1pQk5+hQYWGhxo8fryZNmig0NFTVqlXT8uXLne08V7t27Sqxl++//97r2pE/9uJyuRQZGXneF8CWdhll9dkXad68ufPvzMxMHThwQPHx8efce9WqVRUUFHROvRcUFKhnz546deqUXn755bPWG2O8rqcq6dqq02uKbNq0Sd98840GDBhw1vUAZaWirxsAbNO/f3899thjkqSXXnqpxJrOnTsrKipKs2fPlsfj0alTpxQdHa38/HxJ/w0ef+bUqVPq3LmzJk2aVGxeUeg4H0Uhpk6dOs7Y6V9WJX2BVapUqVh90SmZKVOmaNq0aZo+fbpz109SUpKzneeqpPWWtI/+2EtRP6efHjqbC1lGWXz2RapWrer8OzAw8KL1XlBQoO7duystLU2rVq1ScHCwMy8yMlL5+fnKyspS9erVnfHMzEy1atXKqfnll1+KLffgwYOKiIgoNv7qq6+qWbNmio2NPadtAsoCR3aAMnbnnXcqPz9f+fn5at++fbH5v/32m3bt2qUnn3xS8fHxatSokbKysrxqGjdurK+++srr7p7Nmzd71dx0003auXOn6tSpo/r163u9/vhFeS5yc3M1a9Ys3XbbbapRo4aCg4Pl8Xi0fv16r7qNGzeqUaNG57zcdevWqWvXrnrggQfUtGlT1atXr9g1P/7+/l5HZ0rSuHHjEnu57rrr5Ofnd879XGxl8dmXJCgoSHXq1NGnn35apv0WBZ3vv/9eK1euVGhoqNf82NhYVapUSStWrHDG0tPTtWPHDifstGzZUtnZ2friiy+cms8//1zZ2dlOTZGjR4/q7bff5qgOLjmO7ABlzM/PzzlKUtIXcfXq1RUaGqpZs2apZs2a2rt3r/7f//t/XjW9evXSmDFjNGDAAD355JPas2ePnn/+ea+awYMHa/bs2br//vs1YsQIhYWF6YcfflBycrJmz579pyEgMzNTJ06c0JEjR5SSkqLJkyfr119/1eLFi52aESNG6JlnntG1116rZs2aae7cuUpNTdXChQvPeV/Ur19f7777rjZu3Kjq1atr6tSpysjI8ApMderU0eeff649e/aoWrVqCgkJKbacYcOG6eabb9azzz6rHj16aNOmTZoxY8Y5nXK5lMrisz+TsWPHatCgQQoPD1eHDh105MgRbdiwQUOGDClVrydPntR9992nrVu36sMPP1RhYaEyMjIk/X5Lvr+/v9xutwYMGKBhw4YpNDRUISEhGj58uGJiYtS2bVtJUqNGjXTnnXfq4Ycf1iuvvCJJGjhwoHOB/R+99dZbOnnypHr37l2qnoHSIuwAF8EfTwWcrkKFCkpOTtbjjz+u6OhoNWzYUC+++KLi4uKcmmrVqumDDz7QoEGDdOONN6px48aaNGmSunXr5tR4PB5t2LBBo0aNUvv27ZWXl6fatWvrzjvvVIUKf37QtmHDhnK5XKpWrZrq1aunhIQEDR06VJGRkU7N448/rpycHA0bNkyZmZlq3Lixli5dqgYNGpzzfnjqqaeUlpam9u3bq0qVKho4cKDuvvtuZWdnOzXDhw9X37591bhxY+Xm5iotLa3Ycm666Sa9/fbbevrpp/Xss8+qZs2a+uc//1kuH9Z3oZ/9mfTt21cnTpzQtGnTNHz4cIWFhV3Q3Uz79+/X0qVLJUnNmjXzmrd69Wqnp2nTpqlixYrq3r27cnNzFR8fr3nz5nmFuYULF+rxxx937trq0qWLZsyYUWydr732mu69916vU2LApeAy53JxAAAAwGWKa3YAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsNr/B14iBQcZIPMgAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x ='Made Donation in March 2007',data=df)\n",
    "plt.title('Donation Distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d23a0bb0-533a-4507-bb51-5e01a193addc",
   "metadata": {},
   "source": [
    "#### Observation\n",
    "- Most donors did not donate again.\n",
    "- Fewer donors donated in March 2007.\n",
    "\n",
    "The blood bank faces a challenge because only a small percentage of previous donors returned."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ce0d2182-194d-4239-9bc5-61e24d0c0ffa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Made Donation in March 2007\n",
       "0    76.041667\n",
       "1    23.958333\n",
       "Name: proportion, dtype: float64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Made Donation in March 2007'].value_counts(normalize=True)*100"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f6a01bcc-6a99-4eb5-a16c-208d7941e9f7",
   "metadata": {},
   "source": [
    "- Only about 1 out of every 4 donors returned to donate again."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6d0681d2-6d89-4e38-9614-bd312d2ccb81",
   "metadata": {},
   "source": [
    "#### Months Since Last Donation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f394c018-2b23-4fc5-aaef-87458c2779c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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bx6Ki4vx7LPPVvs48fHxSEtLs/2pPyAgoFpTPL3xxhvo3LkzWrduDb1ej++++67KUL17925s3rwZAwYMQHh4OHbv3o3Lly/XKITfqFevXrjnnnvwyCOPYP78+WjatCmOHTsGSZJw3333Ydq0abjzzjsxfvx4PP/88/Dz80NqaipSUlKwcOHCKvdrMplw8OBBu2UajQYtWrRAaGgolixZgqioKKSnp2P69Ol227333nuIiopChw4doFAo8M033yAyMtI2xVl8fDw2b96MHj16QKvVIjg4uMo6rl27hqysLOj1epw4cQL//ve/sW7dOnz66ae2/b3yyivo2rUr/v73v+Pxxx/Hrl27sGjRInz44YfVPo9NmzZFeno6Vq1aha5du+L777+v8AtHdb4/nnzySbz55psYPXo0kpKScPnyZUycOBGjRo1CREREteshItfjyC0RuVRgYCACAwOrXP/222/jkUcewahRo9CpUyecOnUKGzduvGlw+qtHHnkE9913H/r06YP69evjyy+/rNbrNBoNZsyYgXbt2uGee+6BUqnEqlWrqnwf27dvx/33349mzZrhb3/7G95999063QRg9erV6Nq1K4YPH45WrVrh9ddft40wtmvXDtu2bcPJkydx9913o2PHjpg1a5at37gqhYWF6Nixo93j/vvvh0KhwKpVq7Bv3z60adMGU6ZMwTvvvGP3Wn9/f8ydOxddunRB165dcfbsWfzwww9QKCz/63j33XeRkpJS6ZRuf/X0008jKioKLVq0wEsvvQR/f3/8/vvvGDFihG2bTp064euvv8aqVavQpk0bvPHGG5g9e3aNbvLx4IMPYsqUKZgwYQI6dOiAnTt3YtasWXbbVOf7w9fXFxs3bsTVq1fRtWtXPProo+jbty8WLVpU7VqIyD0kUVlDExERERGRB+LILRERERF5DYZbIiIiIvIaDLdERERE5DUYbomIiIjIazDcEhEREZHXYLglIiIiIq/BcEtEREREXoPhloiIiIi8BsMtEREREXkNhlsiIiIi8hoMt0RERETkNf4/toJfUGPsNGsAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,5))\n",
    "sns.histplot(df['Months since Last Donation'],bins=20,kde=True)\n",
    "plt.title('Months Since Last Donation')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b62016e-d585-488f-8182-59753af69a12",
   "metadata": {},
   "source": [
    "- Lower values indicate recent donors.\n",
    "\n",
    "- Recent donors may have a higher probability of donating again."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e018c2a-aa65-4252-95ec-a36b3902dc1d",
   "metadata": {},
   "source": [
    "#### Number of Donation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "2e3f8891-037f-4f67-96cd-59512a34958b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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JSVF8fHxHn3vcrmUnp5O39eDYRgaN26clixZotWrVyslJcVjPf3dOAzDkNPppL+97Oabb9bWrVu1ZcsW99SrVy/df//92rJli9q1a0d/NzCn06mdO3cqISHB97/fDX4KtAktXrzYCAgIMN566y1jx44dxvjx443Q0FDjwIEDvi6t2SssLDQ2b95sbN682ZBkzJ4929i8ebNx8OBBwzAMY+bMmYbdbjeWLFlibN261bj33nuNhIQEw+Fw+Ljy5uexxx4z7Ha7sXbtWiMnJ8c9lZSUuNvQ3941efJkY926dUZWVpbx/fffG88++6zh5+dnrFixwjAM+ruh/fJqLMOgv71t4sSJxtq1a439+/cbX3/9tTF8+HAjPDzc/d3oy/4m7NTT3LlzjeTkZCMwMNDo0aOH+3JdXJw1a9YYkmpMo0aNMgyj+vLFKVOmGPHx8YbNZjNuvPFGY+vWrb4tupk6Wz9LMubPn+9uQ39710MPPeT+d6Nly5bGzTff7A46hkF/N7Qzww797V0jR440EhISjICAACMxMdG48847je3bt7vX+7K/LYZhGA0/fgQAAOAbnLMDAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbAD4KIcOHBAFotFW7Zs8XUpbrt27VKfPn0UFBSkq666ytflXJTRo0fr9ttv93UZQLNG2AGaudGjR8tisWjmzJkey5ctWyaLxeKjqnxrypQpCg0N1e7du7Vq1aqztjndbxaLRQEBAYqLi9PgwYP19ttvq6qqqpErPndofOWVV7RgwYJGrwcwE8IOYAJBQUGaNWuWTpw44etSvMblctX7vfv27dP111+v5ORkRUdHn7PdsGHDlJOTowMHDuizzz7TwIED9dRTT2n48OGqqKio9+d7k91uV4sWLXxdBtCsEXYAExg0aJDi4+M1Y8aMc7aZOnVqjUM6L7/8stq2beueP33IZPr06YqLi1OLFi00bdo0VVRU6Omnn1ZUVJRat26tt99+u8b2d+3apX79+ikoKEhdunTR2rVrPdbv2LFDv/rVrxQWFqa4uDg98MADOnbsmHv9gAEDNG7cOE2YMEExMTEaPHjwWfejqqpKzz//vFq3bi2bzaarrrpKy5cvd6+3WCzKzMzU888/L4vFoqlTp56zT2w2m+Lj49WqVSv16NFDzz77rD788EN99tlnHqMphw4d0m233aawsDBFRETo7rvv1pEjR2r07b/+9S+1bdtWdrtd99xzjwoLC91tli9fruuvv14tWrRQdHS0hg8frn379rnXp6SkSJKuvvpqWSwWDRgwwOPP5DSn06knn3xSsbGxCgoK0vXXX6+MjAz3+rVr18pisWjVqlXq1auXQkJC1K9fP+3evdvd5rvvvtPAgQMVHh6uiIgI9ezZUxs3bjxnPwHNHWEHMAGr1arp06frtdde048//nhR21q9erUOHz6sdevWafbs2Zo6daqGDx+uyMhIffPNN3r00Uf16KOPKjs72+N9Tz/9tCZOnKjNmzerX79+uvXWW5Wfny9JysnJUf/+/XXVVVdp48aNWr58uY4cOaK7777bYxsLFy6Uv7+/vvrqK73xxhtnre+VV17Riy++qL///e/6/vvvNXToUN16663as2eP+7O6dOmiiRMnKicnR5MmTbqg/b/pppvUvXt3LVmyRJJkGIZuv/12HT9+XOnp6UpLS9O+ffs0cuRIj/ft27dPy5Yt08cff6yPP/5Y6enpHocWi4uLNWHCBGVkZGjVqlXy8/PTHXfc4T5k9u2330qSVq5cqZycHPfnn+kPf/iD/vOf/2jhwoXatGmT2rdvr6FDh+r48eMe7Z577jm9+OKL2rhxo/z9/fXQQw+5191///1q3bq1MjIylJmZqWeeeUYBAQEX1E9As9Ioz1YH0GBGjRpl3HbbbYZhGEafPn2Mhx56yDAMw1i6dKnxy7/iU6ZMMbp37+7x3pdeeslITk722FZycrJRWVnpXtapUyfjhhtucM9XVFQYoaGhxvvvv28YhmFkZWUZkoyZM2e625SXlxutW7c2Zs2aZRiGYfz5z382hgwZ4vHZ2dnZhiRj9+7dhmEYRv/+/Y2rrrrqvPubmJho/PWvf/VYds011xiPP/64e7579+7GlClTat3OL/vtTCNHjjSuuOIKwzAMY8WKFYbVajUOHTrkXr99+3ZDkvHtt98ahlHdtyEhIYbD4XC3efrpp43evXuf8/Pz8vIMScbWrVsNw/i5Hzdv3nzOOouKioyAgABj0aJF7vUul8tITEw0XnjhBcMwDGPNmjWGJGPlypXuNp988okhySgtLTUMwzDCw8ONBQsW1NY9gKkwsgOYyKxZs7Rw4ULt2LGj3tvo0qWL/Px+/qchLi5O3bp1c89brVZFR0crLy/P4319+/Z1v/b391evXr20c+dOSVJmZqbWrFmjsLAw93T55ZdLksehnF69etVam8Ph0OHDh3Xdddd5LL/uuuvcn+UNhmG4T+7euXOnkpKSlJSU5F7fuXNntWjRwuMz27Ztq/DwcPd8QkKCRx/t27dP9913n9q1a6eIiAj3YatDhw7Vua59+/apvLzcY/8DAgJ07bXX1tj/K6+80qMWSe56JkyYoEceeUSDBg3SzJkzPf4MADMi7AAmcuONN2ro0KF69tlna6zz8/OTYRgey8rLy2u0O/Nwxumrlc5cVpcrlk4HhqqqKo0YMUJbtmzxmPbs2aMbb7zR3T40NPS82/zldk/7ZTjxhp07d7rDyLm2feby8/XRiBEjlJ+fr3/+85/65ptv9M0330i6sBOxT//51WX/f1nPL/8cpOpzjLZv365bbrlFq1evVufOnbV06dI61wE0N4QdwGRmzpyp//3f/9X69es9lrds2VK5ubkegceb98b5+uuv3a8rKiqUmZnpHr3p0aOHtm/frrZt26p9+/YeU10DjiRFREQoMTFRX375pcfy9evX64orrvDKfqxevVpbt27VXXfdJal6FOfQoUMe5yjt2LFDBQUFdf7M/Px87dy5U3/60590880364orrqhx5VxgYKAkqbKy8pzbad++vQIDAz32v7y8XBs3brzg/e/YsaN+//vfa8WKFbrzzjs1f/78C3o/0JwQdgCT6datm+6//3699tprHssHDBigo0eP6oUXXtC+ffs0d+5cffbZZ1773Llz52rp0qXatWuXxo4dqxMnTrhPih07dqyOHz+ue++9V99++63279+vFStW6KGHHqr1y/1snn76ac2aNUsffPCBdu/erWeeeUZbtmzRU089dcE1O51O5ebm6qefftKmTZs0ffp03XbbbRo+fLgefPBBSdVXul155ZW6//77tWnTJn377bd68MEH1b9///MedjstMjJS0dHRevPNN7V3716tXr1aEyZM8GgTGxur4OBg98nbBQUFNbYTGhqqxx57TE8//bSWL1+uHTt2aMyYMSopKdHDDz9cp1pKS0s1btw4rV27VgcPHtRXX32ljIwMr4VFoCki7AAm9Je//KXGIasrrrhC8+bN09y5c9W9e3d9++23F3ylUm1mzpypWbNmqXv37vriiy/04YcfKiYmRpKUmJior776SpWVlRo6dKi6du2qp556Sna73eP8oLp48sknNXHiRE2cOFHdunXT8uXL9dFHH6lDhw4XXPPy5cuVkJCgtm3batiwYVqzZo1effVVffjhh7JarZKqDwEtW7ZMkZGRuvHGGzVo0CC1a9dOH3zwQZ0/x8/PT4sXL1ZmZqa6du2q3//+9/rb3/7m0cbf31+vvvqq3njjDSUmJuq2224767Zmzpypu+66Sw888IB69OihvXv36vPPP1dkZGSdarFarcrPz9eDDz6ojh076u6771ZqaqqmTZtW5/0BmhuLcea/iAAAACbCyA4AADA1wg4AADA1wg4AADA1wg4AADA1wg4AADA1wg4AADA1wg4AADA1wg4AADA1wg4AADA1wg4AADA1wg4AADC1/w8Gr72P3OEOdwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(df['Number of Donations'],kde=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee470dbe-6d4e-4d55-ae8d-b0f759ea2459",
   "metadata": {},
   "source": [
    "- Frequent donors are generally more likely to donate again."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ddc6516-d8f3-41e3-8f46-af711619cba2",
   "metadata": {},
   "source": [
    "#### Correlation Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c42d3834-1e2a-49bc-9a50-20a4dd9bef56",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0AAAALCCAYAAAD3bLMlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAADMxElEQVR4nOzdd1gUx/8H8PfRe5cqAooKiICAiYg1Fuw91p+KGhNFRYNdY01iixo7lq+KJsaSWGMUxV6wgaJGsCGIBUSkKnCUu98fhMOTQzkV4bz363n20Zud3Z3ZhWNnPzOzArFYLAYREREREZESUKnsAhAREREREX0qbAAREREREZHSYAOIiIiIiIiUBhtARERERESkNNgAIiIiIiIipcEGEBERERERKQ02gIiIiIiISGmwAUREREREREqDDSAiIiIiIlIabAAREREREZHSYAOIiIiIiIg+2JkzZ9C5c2dYW1tDIBBg375979zm9OnT8PLygpaWFmrWrIm1a9dWeDnZACIiIiIiog/26tUruLu7Y9WqVeXKHxcXhw4dOqBp06a4du0apk2bhsDAQOzevbtCyykQi8XiCj0CEREREREpFYFAgL1796Jbt25l5pk8eTIOHDiAmJgYSdqIESNw/fp1XLhwocLKxggQERERERHJJBQKkZmZKbUIhcKPsu8LFy6gbdu2Uml+fn6IiIhAfn7+RzmGLGoVtmciei//qNet7CLQfzYHHK7sIhBVSYMG2lV2Eeg/i+acq+wi0H/O/d280o5dkfcOV6b3w5w5c6TSZs2ahdmzZ3/wvpOSkmBhYSGVZmFhgYKCAqSkpMDKyuqDjyELG0BERERERCTT1KlTERQUJJWmqan50fYvEAikPhePznkz/WNiA4iIiIiISIEJ1CuusaCpqflRGzyvs7S0RFJSklRacnIy1NTUYGpqWiHHBDgGiIiIiIiIKoGPjw/CwsKk0o4ePQpvb2+oq6tX2HHZACIiIiIiUmAqaoIKW+Tx8uVLREVFISoqCkDRNNdRUVFISEgAUNSdbtCgQZL8I0aMwMOHDxEUFISYmBhs2rQJGzduxIQJEz7auZGFXeCIiIiIiBSYQL1qxDQiIiLQsmVLyefisUODBw9GSEgIEhMTJY0hAHBwcMChQ4fw/fffY/Xq1bC2tsaKFSvQs2fPCi0nG0BERERERPTBWrRogbe9YjQkJKRUWvPmzXH16tUKLFVpbAARERERESkwebuqKbuqES8jIiIiIiL6BBgBIiIiIiJSYBU5DfbniBEgIiIiIiJSGowAEREREREpMI4Bkg8jQEREREREpDQYASIiIiIiUmAcAyQfNoCIiIiIiBQYu8DJh13giIiIiIhIaTACRERERESkwASqjADJgxEgIiIiIiJSGowAEREREREpMBVGgOTCCBARERERESkNRoCIiIiIiBSYQIURIHkwAkREREREREqDESAiIiIiIgUmUGVMQx5sABERERERKTBOgiAfNheJiIiIiEhpMAJERERERKTAOAmCfBgBIiIiIiIipcEIEBERERGRAuMYIPkwAkREREREREqDESAiIiIiIgUmYARILowAERERERGR0mAEiIiIiIhIgQlUGNOQBxtAREREREQKjNNgy4fNRSIiIiIiUhqMABERERERKTBOgy0fRoCIiIiIiEhpMAJERERERKTAOAZIPowAERERERGR0mAEiIiIiIhIgXEabPnwbBERERERkdJQ+gaQQCDAvn37Psmx/P390a1bt09yLPo4PuXPBxEREdH7EKgIKmz5HFVKFzh/f39s2bIF3333HdauXSu1LiAgAMHBwRg8eDBCQkI+2jFnz56Nffv2ISoq6qPtU17Lly+HWCyutOMDQEhICMaNG4f09PQK2b9AIMDevXvf2dATCEp+oXR0dGBtbQ1fX1+MGTMGXl5eFVK2tynr5yMxMRHGxsafvDyKyqSJN2qOHwZDT1doWZsjomcAnh04XtnF+iz1bmeM1o31oautgvsPhdjwVwoeJ+WXmb+6pTr6djBBzeoaMDdVx+Y9KfjndGapfCaGqvi/LiZo4KwDDXUBnibnI3j7czx4nFeR1VFovBZVQ3jYdpz6ZxOy0p/DwsYRXQZOQU0nb5l5M9Oe4+9ti/Ak/hZSkh7C1+//0HXgVKk8hQX5OHFgAyLO7kdm2jNUs3JAh75BcHJv+imqo/CG9rNDFz8r6OupIfpuFpauvYe4hOwy83dua4l2X1mipp0OAODO/ZdYtzUOMfeypPKZmWhgpH9NNPIygaamCh49ycGCFXdwJ/ZlhdanquM02PKptAiQra0tduzYgZycHElabm4utm/fjho1alRWsSqUoaEhjIyMKrsYVcbmzZuRmJiIW7duYfXq1Xj58iW+/PJLbN26tbKLJmFp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LDo6GoGBgZg5cyYiIyMRGhqKuLg4jBgx4qOdnzdxGmwiIiIiIpJJU1MTmpqa5cprZmYGVVXVUtGe5OTkUlGhYvPnz4evry8mTpwIAHBzc4Ouri6aNm2Kn376CVZWVh9WARkYASIiIiIiUmAVOQZIHhoaGvDy8kJYWJhUelhYGBo3bixzm+zsbKi8cRxVVVUARZGjisAGEBERERERfRRBQUH43//+h02bNiEmJgbff/89EhISJF3apk6dikGDBknyd+7cGXv27EFwcDAePHiA8+fPIzAwEF988QWsra0rpIzsAkdEREREpMDe5309FaVPnz548eIF5s6di8TERLi6uuLQoUOws7MDACQmJkq9E8jf3x9ZWVlYtWoVxo8fDyMjI3z11VdYuHBhhZWRDSAiIiIiIgVWlRpAABAQEICAgACZ60JCQkqljRkzBmPGjKngUpVgFzgiIiIiIlIajAARERERESkyOScrUHY8W0REREREpDQYASIiIiIiUmBlvWSUZGMEiIiIiIiIlAYjQERERERECkzeF5YqO54tIiIiIiJSGowAEREREREpsKr2HqCqjg0gIiIiIiJFxi5wcuHZIiIiIiIipcEIEBERERGRAmMXOPkwAkREREREREqDESCiKmZzwOHKLgL9Z8ia9pVdBKIqqVWzEZVdBPqP9k8BlV0EqgIEAsY05MGzRURERERESoMRICIiIiIiRcYxQHJhBIiIiIiIiJQGI0BERERERApMwPcAyYUNICIiIiIiBcZpsOXD5iIRERERESkNRoCIiIiIiBQZp8GWC88WEREREREpDUaAiIiIiIgUGMcAyYcRICIiIiIiUhqMABERERERKTJOgy0Xni0iIiIiIlIajAARERERESkwgYBjgOTBBhARERERkSJjFzi58GwREREREZHSYASIiIiIiEiBcRps+TACRERERERESoMRICIiIiIiRSZgTEMePFtERERERKQ0GAEiIiIiIlJkHAMkF0aAiIiIiIhIaTACRERERESkwAQcAyQXNoCIiIiIiBQZu8DJhc1FIiIiIiJSGowAEREREREpMIEKYxry4NkiIiIiIiKlwQgQEREREZEiE3AMkDwYASIiIiIiIqXBCBARERERkSLjGCC58GwREREREZHSYASIiIiIiEiRcQyQXNgAIiIiIiJSYJwGWz48W0REREREpDQYASIi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Q1hYGMaMGQNvb2+Ympqid+/e+OmnnyqsjGwAEREREREpsir0IlQACAgIQEBAgMx1ISEhpdKcnJxKdZurSOwCR0RERERESoMRICIiIiIiRSbnZAXKjmeLiIiIiIiUBiNARERERESKrIqNAarqGAEiIiIiIiKlwQgQEREREZEie4/39Sgzni0iIiIiIlIajAARERERESkwMccAyYUNICIiIiIiRcZpsOXCs0VEREREREqDESAiIiIiIkXGCJBceLaIiIiIiEhpMAJERERERKTAOAmCfBgBIiIiIiIipcEIEBERERGRIuMYILnwbBERERERkdJgBIiIiIiISJFxDJBc2AAiIiIiIlJkKuzUJQ+eLSIiIiIiUhqMABERERERKTBOgy0fRoCIiIiIiEhpMAJERERERKTIOA22XHi2iIiIiIhIaTACRERERESkwMSMAMmFZ4uIiIiIiJQGI0BERERERIqMs8DJhQ0gIiIiIiIFxi5w8uHZIiIiIiIipcEIEBERERGRImMXOLkwAkREREREREqDESAiIiIiIkXGMUBy4dkiIiIiIiKlwQgQEREREZECE3MMkFwYASIiIiIiIqXBCBARERERkSLjGCC5sAFERERERKTAxGAXOHmwuUhEREREREqDESAiIiIiIgUmZhc4ufBsERERERGR0mAEiIiIiIhIkTECJBeeLSIiIiIiUhqMABERERERKTC+CFU+jAAREREREZHSYASIiIiIiEiBcRY4+bABRERERESkyNgFTi5sLhIRERERkdJgBIiIiIiISIGxC5x8eLaIiIiIiEhpMAJERERERKTAxOAYIHkwAkREREREREqDESAiIiIiIgXGMUDy4dkiIiIiIiKlwQgQEREREZEi43uA5MIGEBERERGRAhOzU5dceLaIiIiIiEhpMAJERERERKTAxOwCJxdGgIiIiIiISGkwAkREREREpMA4DbZ8eLaIiIiIiEhpMAJERERERKTAxOAYIHkwAkREREREREqDESAiIiIiIgXGMUDy4dkiIiIiIlJgYoGgwpb3sWbNGjg4OEBLSwteXl44e/ZsubY7f/481NTU4OHh8V7HLS82gIiIiIiI6KPYuXMnxo0bh+nTp+PatWto2rQp2rdvj4SEhLdul5GRgUGDBqFVq1YVXkY2gIiIiIiIFJgYggpb5LV06VIMGzYM33zzDZydnbFs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dXD06FFMmzYNX3zxBbS1tfHll1+iX79+AICUlBSpcVXy5geA7du3Y8CAAVJjlhTVrtA07Aote8rv4kZNseepBeg19t3Tf0feykbkLT7F+9hSz1zGP+p8V0lVwGvxafRp5Io+jVxlrvux11el0hzMjbFuaNnv+Ds8SbHf31aZmvn1RTO/vjLXFTdqio2bXfoVJG+auzr0o5TrcyauWp26qjyB+H0HVRApmDVr1mD//v04cuTIJzne8+fP4eTkhIiICLm6J5an0UCfxpA17Su7CERVUqsdIyq7CPSfs46MgFQVbdwrrwv73djyT/wlrzq1alTYvitLlYoAEVWkb7/9FmlpacjKypKawa6ixMXFYc2aNe81NouIiIiIKgYbQKQ01NTUMH369E92vC+++AJffPHFuzMSERERfYDP9YWlFYUdBomIiIiISGkwAkREREREpMAYAZIPI0BERERERKQ0GAEiIiIiIlJgjADJhxEgIiIiIiJSGowAEREREREpMLGYESB5sAFERERERKTA2AVOPuwCR0RERERESoMRICIiIiIiBcYIkHwYASIiIiIiIqXBCBARERERkQJjBEg+jAAREREREZHSYASIiIiIiEiBcRps+TACRERERERESoMRICIiIiIiBSbiGCC5sAFERERERKTAOAmCfNgFjoiIiIiIlAYjQERERERECoyTIMiHESAiIiIiIlIajAARERERESkwjgGSDyNARERERESkNBgBIiIiIiJSYBwDJB9GgIiIiIiISGkwAkREREREpMA4Bkg+bAARERERESkwdoGTD7vAERERERGR0mAEiIiIiIhIgYkquwAKhhEgIiIiIiJSGowAEREREREpMI4Bkg8jQEREREREpDQYASIiIiIiUmCcBls+jAAREREREZHSYASIiIiIiEiBcQyQfNgAIiIiIiJSYOwCJx92gSMiIiIiIqXBCBARERERkQITiSu7BIqFESAiIiIiIlIajAARERERESkwjgGSDxtARFXMoIF2lV0E+k+rZiMquwhEVdLxvmsruwj0n/NrAyq7CPSfNu6VXQIqLzaAiIiIiIgUGKfBlg/HABERERERkdJgBIiIiIiISIGJOQucXNgAIiIiIiJSYCJOgiAXdoEjIiIiIiKlwQgQEREREZEC4yQI8mEEiIiIiIiIlAYjQERERERECoyTIMiHESAiIiIiIlIajAARERERESkwMWeBkwsjQEREREREpDTYACIiIiIiUmAiccUt72PNmjVwcHCAlpYWvLy8cPbs2TLz7tmzB23atEG1atVgYGAAHx8fHDly5D3PRPmwAUREREREpMDEYkGFLfLauXMnxo0bh+nTp+PatWto2rQp2rdvj4SEBJn5z5w5gzZt2uDQoUOIjIxEy5Yt0blzZ1y7du1DT0uZBGIx540gqkoORBRWdhHoP20frqjsIhBVScf7rq3sItB/zq+9UdlFoP/MG6ZZacc+fC2/wvbdvoG6XPm//PJLeHp6Ijg4WJLm7OyMbt26Yf78+eXaR7169dCnTx/MnDlTrmOXFydBICIiIiJSYBUZzhAKhRAKhVJpmpqa0NQs3eDLy8tDZGQkpkyZIpXetm1bhIeHl+t4IpEIWVlZMDExef9CvwO7wBERERERkUzz58+HoaGh1FJWJCclJQWFhYWwsLCQSrewsEBSUlK5jrdkyRK8evUKvXv3/uCyl4URICIiIiIiBSaqwGmwp06diqCgIKk0WdGf1wkE0uURi8Wl0mTZvn07Zs+ejf3798Pc3Fz+wpYTG0BERERERCRTWd3dZDEzM4OqqmqpaE9ycnKpqNCbdu7ciWHDhuHPP/9E69at37u85cEucERERERECkwsrrhFHhoaGvDy8kJYWJhUelhYGBo3blzmdtu3b4e/vz/++OMPdOzY8X1OgVwYASIiIiIioo8iKCgIAwcOhLe3N3x8fLB+/XokJCRgxIgRAIq61D158gRbt24FUNT4GTRoEJYvX45GjRpJokfa2towNDSskDKyAUREREREpMDe5309FaVPnz548eIF5s6di8TERLi6uuLQoUOws7MDACQmJkq9E2jdunUoKCjAqFGjMGrUKEn64MGDERISUiFlZAOIiIiIiEiBiarYWz0DAgIQEBAgc92bjZpTp05VfIHewDFARERERESkNBgBIiIiIiJSYBX5ItTPESNARERERESkNBgBIiIiIiJSYOIKfBHq54gRICIiIiIiUhqMABERERERKbCqNgtcVccIEBERERERKQ1GgIiIiIiIFBhngZMPI0BERERERKQ0GAEiIiIiIlJgjADJhw0gIiIiIiIFJhJzGmx5sAscEREREREpDUaAiIiIiIgUGLvAyYcRICIiIiIiUhqMABERERERKTBGgOTDCBARERER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w4cPx4ULF7Bx40Zs3769wsrIBhAREREREX0Uffr0wYsXLzB37lwkJibC1dUVhw4dgp2dHQAgMTFR6p1ADg4OOHToEL7//nusXr0a1tbWWLFiBXr27FlhZWQDiIiIiIhIgb3H+0orVEBAAAICZEeKQ0JCSqU1b94cV69ereBSlWADiIiIiIhIgYmqUBc4RcBJEIiIiIiISGkwAkREREREpMDEVa0PXBXHCBARERERESkNRoCIiIiIiBSYWFTZJVAsjAAREREREZHSYASIiIiIiEiBiTgGSC6MABERERERkdJgBIiIiIiISIFxFjj5sAFERERERKTA+CJU+bALHBERERERKQ1GgIiIiIiIFBh7wMmHESAiIiIiIlIajAARERERESkwMccAyYURICIiIiIiUhqMABERERERKTC+CFU+jAAREREREZHSYASIiIiIiEiBcQyQfBgBIiIiIiIipcEIEBERERGRAmMESD5sABERERERKTC2f+TDLnBERERERKQ0GAEiIiIiIlJg7AInH0aAiIiIiIhIaTACRERERESkwMR8EapcGAEiIiIiIiKlwQgQEREREZECE3EMkFwYASIiIiIiIqXBCBARERERkQLjGCD5sAFERERERKTAOA22fNgFjoiIiIiIlAYjQERERERECowRIPkwAkREREREREqDESAiIiIiIgUm4iQIcmEEiIiIiIiIlAYjQERERERECoxjgORT5SJA8fHxEAgEiIqKquyiVLhTp05BIBAgPT29Qo/TokULjBs3rkKP8akIBALs27evsotBRERERApKrgiQv78/tmzZgu+++w5r166VWhcQEIDg4GAMHjwYISEhH7OMcjt16hRatmwJoOiGWV9fHzVr1kSbNm3w/fffw8rK6pOXqUWLFvDw8MCyZcskaY0bN0ZiYiIMDQ0r9Nh79uyBurr6B+1DUa79m/Lz8/HDDz/g0KFDePDgAQwNDdG6dWssWLAA1tbWknxCoRATJkzA9u3bkZOTg1atWmHNmjWoXr26JE9aWhoCAwNx4MABAECXLl2wcuVKGBkZAQBCQkIwZMgQmeV49uwZzM3NK66in8DQfnbo4mcFfT01RN/NwtK19xCXkF1m/s5tLdHuK0vUtNMBANy5/xLrtsYh5l6WVD4zEw2M9K+JRl4m0NRUwaMnOViw4g7uxL6s0PooqjNHduD4gRBk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",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(df.corr(),annot=True,cmap='coolwarm')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c79e1494-c241-4404-a97a-9100ae45e8cc",
   "metadata": {},
   "source": [
    "- The heatmap helps identify relationships among donor attributes and the target variable."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ace110d1-5a90-4ebd-91ac-f43dbbccfdf2",
   "metadata": {},
   "source": [
    "#### Feature Selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "84438f46-598e-4f1c-beb4-bd0ae0880481",
   "metadata": {},
   "outputs": [],
   "source": [
    "X= df.drop('Made Donation in March 2007',axis=1)\n",
    "y=df['Made Donation in March 2007']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "5c474c2c-0be4-4990-8123-a17bad771708",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(576, 4)\n",
      "(576,)\n"
     ]
    }
   ],
   "source": [
    "print(X.shape)\n",
    "print(y.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "75572757-9420-4420-90b4-772bef6266a7",
   "metadata": {},
   "source": [
    "The independent variables contain donor-related information.\n",
    "\n",
    "The dependent variable indicates whether the donor donated blood in March 2007."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b5c79f8-f73f-4ab2-9b33-2abc9980825c",
   "metadata": {},
   "source": [
    "#### Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "784a863f-dfb0-49bf-9d22-8e48b4c93835",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3051fdd3-f99c-41b8-b992-fa63bdcdc0b9",
   "metadata": {},
   "source": [
    "80% of the data is used for training and 20% for testing.\n",
    "\n",
    "This helps evaluate the model on unseen data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "02527d8b-de3f-4e86-82be-e1b058c30172",
   "metadata": {},
   "source": [
    "#### Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "3cb4d051-5bcd-4777-98fd-14ead6e6b9dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled =scaler.fit_transform(X_train)\n",
    "X_test_scaled =scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7cf4273-ea36-4856-9c1d-f1b172d988af",
   "metadata": {},
   "source": [
    "- Feature scaling standardizes all variables and improves model performance."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ddf34a06-f0ca-4911-b50e-e360e97bf0ac",
   "metadata": {},
   "source": [
    "#### Model Building"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84f4a797-4203-43af-826b-86456850c106",
   "metadata": {},
   "source": [
    "#### 1. Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "12ba594d-0d4f-427c-8f9a-18cf17de19a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7586206896551724\n"
     ]
    }
   ],
   "source": [
    "lr = LogisticRegression()\n",
    "lr.fit(X_train_scaled, y_train)\n",
    "pred_lr = lr.predict(X_test_scaled)\n",
    "acc_lr = accuracy_score(\n",
    "    y_test,pred_lr)\n",
    "print(acc_lr)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2179c91-6972-49b0-934d-6c02a24978da",
   "metadata": {},
   "source": [
    "#### 2. Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "d402366f-9154-4964-9fee-1dad0307fa37",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.6810344827586207\n"
     ]
    }
   ],
   "source": [
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(X_train,y_train)\n",
    "pred_dt = dt.predict(X_test)\n",
    "acc_dt = accuracy_score(y_test,pred_dt)\n",
    "print(acc_dt)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9891be03-b526-4847-8896-5f146294836b",
   "metadata": {},
   "source": [
    "#### 3. Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "2f06068e-a4f1-494f-a67b-fcd5190f5da0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7155172413793104\n"
     ]
    }
   ],
   "source": [
    "rf= RandomForestClassifier(random_state=42)\n",
    "rf.fit(X_train,y_train)\n",
    "pred_rf = rf.predict(X_test)\n",
    "acc_rf = accuracy_score(y_test,pred_rf)\n",
    "print(acc_rf)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f9bb0168-a6a6-469b-8cdd-f1db4f208ff9",
   "metadata": {},
   "source": [
    "#### 4. KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f2df02d4-deff-4554-9548-a12d3da95cbf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8017241379310345\n"
     ]
    }
   ],
   "source": [
    "knn = KNeighborsClassifier()\n",
    "knn.fit(X_train_scaled,y_train)\n",
    "pred_knn = knn.predict(X_test_scaled)\n",
    "acc_knn = accuracy_score(y_test,pred_knn)\n",
    "print(acc_knn)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7178cf4e-ec2a-42f3-8f86-24694ac3486d",
   "metadata": {},
   "source": [
    "#### 5. SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "e1eb67b3-5889-49b6-9ccb-e1a2cfcb1583",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7586206896551724\n"
     ]
    }
   ],
   "source": [
    "svm = SVC()\n",
    "svm.fit(X_train_scaled,y_train)\n",
    "pred_svm = svm.predict(X_test_scaled)\n",
    "acc_svm = accuracy_score(y_test,pred_svm)\n",
    "print(acc_svm)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b9483f4-58d5-4cbe-9feb-84d4fd0a1e7d",
   "metadata": {},
   "source": [
    "#### Model Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "7c9e6b73-e284-4bbd-be85-ec3e87342bf7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Accuracy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>KNN</td>\n",
       "      <td>0.801724</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Logistic Regression</td>\n",
       "      <td>0.758621</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>SVM</td>\n",
       "      <td>0.758621</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.715517</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Decision Tree</td>\n",
       "      <td>0.681034</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Model  Accuracy\n",
       "3                  KNN  0.801724\n",
       "0  Logistic Regression  0.758621\n",
       "4                  SVM  0.758621\n",
       "2        Random Forest  0.715517\n",
       "1        Decision Tree  0.681034"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results = pd.DataFrame({\n",
    "    'Model':['Logistic Regression',\n",
    "             'Decision Tree',\n",
    "             'Random Forest',\n",
    "             'KNN',\n",
    "             'SVM'\n",
    "            ],\n",
    "    'Accuracy':[\n",
    "        acc_lr,\n",
    "        acc_dt,\n",
    "        acc_rf,\n",
    "        acc_knn,\n",
    "        acc_svm\n",
    "    ]\n",
    "})\n",
    "results.sort_values(\n",
    "    by= 'Accuracy',\n",
    "    ascending=False\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55f4b9a5-cdd1-4058-a004-9255c059545f",
   "metadata": {},
   "source": [
    "#### Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "04858f56-46af-4d26-849e-685a881461fe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAyIAAAHUCAYAAADGGZxzAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABKaklEQVR4nO3deXxN1/7/8feJyDyZBSERNdZQRataimhcFBc1VA01tUUp91bRXtHBUC1Kq6VK0qp5bKrGGlKlraGCVm5IRHFRLZUYaois3x9+zrdHIpI02Rm8no/Hfpx71l577c9eJ73nvO29z7EZY4wAAAAAwEJOuV0AAAAAgHsPQQQAAACA5QgiAAAAACxHEAEAAABgOYIIAAAAAMsRRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBACQrSIiImSz2WSz2bR169ZU640xqlSpkmw2mx5//PFs3bfNZtPYsWMzvd3Ro0dls9kUERGRof6//vqrRo4cqZo1a8rLy0tubm667777NHToUB0+fDjT+89vbr3GR48eze1SAORjzrldAACgYPL29tacOXNShY2oqCjFx8fL29s7dwr7m3bu3Kk2bdrIGKPBgwerYcOGcnFxUWxsrD7//HM1aNBAf/zxR26XmaNat26t7777Tv7+/rldCoB8jCACAMgRXbp00fz58zVjxgz5+PjY2+fMmaOGDRsqKSkpF6vLmqSkJLVr105ubm7asWOHypUrZ1/3+OOP67nnntOyZctyscKc9eeff8rNzU0lSpRQiRIlcrscAPkcl2YBAHJEt27dJEkLFy60tyUmJmr58uXq06dPmtucO3dOAwcOVNmyZeXi4qKKFSvq1Vdf1dWrVx36JSUlqX///ipWrJi8vLzUsmVLHTp0KM0xDx8+rKefflolS5aUq6urqlWrphkzZmTpmGbPnq3Tp09r0qRJDiHkrzp16uTwPDIyUg0bNpSHh4e8vb3VokULfffddw59xo4dK5vNpv379+upp56Sr6+vihYtquHDhys5OVmxsbFq2bKlvL29FRgYqEmTJjlsv3XrVtlsNn3++ecaPny4SpcuLXd3dzVp0kR79+516Lt792517dpVgYGBcnd3V2BgoLp166ZffvnFod+ty682bNigPn36qESJEvLw8NDVq1fTvDRr7969atOmjX2ey5Qpo9atW+vEiRP2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Z1W9enW5urrq2LFjaV6GBQAAgHsLQeQurl69qtOnT+vGjRv69ddftW7dOk2YMEFt2rRRz549JUkhISFq2LCh2rdvr7fffltVqlTRyZMntWbNGrVv31716tVTWFiYmjdvruDgYHXt2lXJyclau3atRowYkWqfERERunHjhh566CF5eHho3rx5cnd3V4UKFVL1DQkJUa1atdS9e3e99957Sk5O1sCBA9WkSZM73gyfEwYNGqTZs2erW7duevnll1W8eHHFxcVp0aJFmj17try9vfXvf/9bw4YNU0pKih599FElJSVpx44d8vLyUq9evSyrFQAAALmPe0TuYt26dfL391dgYKBatmypLVu2aPr06friiy9UqFAhSTcvc1qzZo0aN26sPn36qHLlyuratauOHj2qUqVKSZIef/xxLV26VJGRkapTp46aNWumH374Ic19+vn5afbs2WrUqJH9TMqXX36pYsWKpeprs9m0atUqFSlSRI0bN1ZISIgqVqyoxYsX59ykpKFMmTLavn27bty4odDQUN1///0aOnSofH195eR088/szTff1JgxYzRhwgRVq1ZNoaGh+vLLL+2XuAEAAODeYTPGmNwuAsiIpKQk+fr6KjExUT4+Ppbsc8qUKYqMjFTbtm01fPhwS/YJAACQX2Xm8xpnRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAADAcgQRIB3BwcEOjwAAAMgeBBEgHR4eHg6PAAAAyB4EEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAAAAgOUIIgAAAAAsRxAB0hEUFKTatWsrKCgot0sBAAAoUGzGGJPbRQAZkZSUJF9fXyUmJsrHxye3ywEAAMBtMvN5jTMiAAAAACxHEAEAAABgOYIIAAAAAMsRRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAADAcgQRAMjD4uLiNHToUMXFxeV2KQAAZCuCCADkYQkJCdq3b58SEhJyuxQAALIVQQQAAACA5QgiAAAAACxHEAEAAABgOYIIAAAAAMsRRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFjOOaMdp0+fnuFBhwwZkqViAACOLl++7PAIAEBBkeEgMnXq1Az1s9lsBBEAyCbx8fEOjwAAFBQZDiIJCQk5WQcAAACAe8jfukfk2rVrio2NVXJycnbVAwAAAOAekKUgcvnyZfXt21ceHh6qUaOGjh07JunmvSETJ07M1gIBAAAAFDxZCiKjRo3Svn37tHXrVrm5udnbQ0JCtHjx4mwrDgAAAEDBlOF7RP5q1apVWrx4sR5++GHZbDZ7e/Xq1bmhEgAAAMBdZemMyG+//aaSJUumar906ZJDMAEAAACAtGQpiNSvX19fffWV/fmt8DF79mw1bNgweyoDAAAAUGBl6dKsCRMmqGXLljp48KCSk5M1bdo0/fzzz/ruu+8UFRWV3TUCAAAAKGCydEbkkUce0fbt23X58mUFBwdrw4YNKlWqlL777js9+OCD2V0jAAAAgAImS2dEJKlmzZr69NNPs7OWAi8wMFAvvfSSXnrppdwuBQAAAMhVGT4jkpSUlOElr+rdu7dsNptsNpucnZ1Vvnx5vfDCC/rjjz9yu7QcNXbsWPtx/3X5+uuvc7WmOnXq5Nr+AQAAkLsyfEbEz88vw9+IdePGjSwXlNNatmyp8PBwJScn6+DBg+rTp4/Onz+vhQsX5nZpOapGjRqpgkfRokWzNNa1a9fk4uKSHWUBAADgHpXhMyJbtmzR5s2btXnzZs2dO1clS5bUiBEjtHLlSq1cuVIjRoxQqVKlNHfu3Jys929zdXVV6dKlVa5cOT3xxBPq0qWLNmzYYF9/48YN9e3bV0FBQXJ3d1eVKlU0bdo0hzF69+6t9u3b691335W/v7+KFSumQYMG6fr16/Y+Z86c0ZNPPil3d3cFBQVp/vz5qWo5duyY2rVrJy8vL/n4+Khz58769ddf7etvnTWYO3euypcvLy8vL73wwgu6ceOGJk2apNKlS6tkyZIaN27cXY/b2dlZpUuXdlhuhYkDBw6oWbNmcnd3V7FixTRgwABdvHgx1fFOmDBBZcqUUeXKlSVJ//vf/9SlSxcVKVJExYoVU7t27XT06FH7dlu3blWDBg3k6ekpPz8/NWrUSL/88osiIiL0+uuva9++ffazMxEREXc9BgAAABQcGT4j0qRJE/v/fuONNzRlyhR169bN3ta2bVvVrFlTH3/8sXr16pW9VeaQI0eOaN26dSpcuLC9LSUlReXKldOSJUtUvHhx7dixQwMGDJC/v786d+5s77dlyxb5+/try5YtiouLU5cuXVSnTh31799f0s0P78ePH9fmzZvl4uKiIUOG6MyZM/btjTFq3769PD09FRUVpeTkZA0cOFBdunTR1q1b7f3i4+O1du1arVu3TvHx8erUqZMSEhJUuXJlRUVFaceOHerTp4+aN2+uhx9+ONNzcPnyZbVs2VIPP/ywdu3apTNnzqhfv34aPHiwQzjYtGmTfHx8tHHjRhljdPnyZTVt2lSPPfaYvvnmGzk7O+utt95Sy5YttX//fjk5Oal9+/bq37+/Fi5cqGvXrmnnzp2y2Wzq0qWLfvrpJ61bt85+lsbX1zdVbVevXtXVq1ftz/PyZX8AAADIJJMF7u7u5tChQ6naY2Njjbu7e1aGtESvXr1MoUKFjKenp3FzczOSjCQzZcqUdLcbOHCg6dixo8M4FSpUMMnJyfa2p556ynTp0sUYc3MeJJnvv//evj4mJsZIMlOnTjXGGLNhwwZTqFAhc+zYMXufn3/+2UgyO3fuNMYYExYWZjw8PExSUpK9T2hoqAkMDDQ3btywt1WpUsVMmDDhjvWHhYUZJycn4+npaV/q169vjDHm448/NkWKFDEXL1609//qq6+Mk5OTOX36tP14S5UqZa5evWrvM2fOHFOlShWTkpJib7t69apxd3c369evN2fPnjWSzNatW+9YU+3ate9Y860+t16jvy6JiYnpbgcUJJMnTzZNmjQxkydPzu1SAAC4q8TExAx/XsvS1/cGBARo5syZqdpnzZqlgICALAUiqzRt2lTR0dH64Ycf9OKLLyo0NFQvvviiQ5+ZM2eqXr16KlGihLy8vDR79mwdO3bMoU+NGjVUqFAh+3N/f3/7GY+YmBg5OzurXr169vVVq1aVn5+f/XlMTIwCAgIc5qt69ery8/NTTEyMvS0wMFDe3t7256VKlVL16tXl5OTk0PbXsy1pqVKliqKjo+3L8uXL7XXUrl1bnp6e9r6NGjVSSkqKYmNj7W01a9Z0uC9kz549iouLk7e3t7y8vOTl5aWiRYvqypUrio+PV9GiRdW7d2+FhobqySef1LRp03Tq1Kl0a7zdqFGjlJiYaF+OHz+eqe0BAACQd2Xp63unTp2qjh07av369fbLgb7//nvFx8fbP+DmVZ6enqpUqZIkafr06WratKlef/11vfnmm5KkJUuWaNiwYZo8ebIaNmwob29vvfPOO/rhhx8cxvnr5VzSzV+XT0lJkXTzsqtbbXdijElz/e3tae0nvX3fiYuLi/24M1LH7fX/NahINy9he/DBB9O896VEiRKSpPDwcA0ZMkTr1q3T4sWL9dprr2njxo0ZvoTM1dVVrq6uGeoLAACA/CVLZ0RatWqlw4cPq23btjp37pzOnj2rdu3a6dChQ2rVqlV215ijwsLC9O677+rkyZOSpG3btumRRx7RwIED9cADD6hSpUqKj4/P1JjVqlVTcnKydu/ebW+LjY3V+fPn7c+rV6+uY8eOOfwr/8GDB5WYmKhq1ar9vYPKhOrVqys6OlqXLl2yt23fvl1OTk72m9LTUrduXR0+fFglS5ZUpUqVHJa/3u/xwAMPaNSoUdqxY4fuv/9+LViwQNLNYJSXv10NAAAAOStLQUSSypUrp/Hjx2vFihVauXKlxo0bl+cvy0rL448/rho1amj8+PGSpEqVKmn37t1av369Dh06pP/85z/atWtXpsasUqWKWrZsqf79++uHH37Qnj171K9fP7m7u9v7hISEqFatWurevbt+/PFH7dy5Uz179lSTJk0cLunKad27d5ebm5t69eqln376SVu2bNGLL76oHj16qFSpUuluV7x4cbVr107btm1TQkKCoqKiNHToUJ04cUIJCQkaNWqUvvvuO/3yyy/asGGDDh06ZA9ZgYGBSkhIUHR0tH7//XeHm9IBAABQ8GU5iJw/f16TJ09Wv3791L9/f02dOlWJiYnZWZtlhg8frtmzZ+v48eN6/vnn1aFDB3Xp0kUPPfSQzp49q4EDB2Z6zPDwcAUEBKhJkybq0KGDBgwYoJIlS9rX22w2rVq1SkWKFFHjxo0VEhKiihUravHixdl5aHfl4eGh9evX69y5c6pfv746deqk5s2b64MPPrjrdt98843Kly+vDh06qFq1aurTp4/+/PNP+fj4yMPDQ//973/VsWNHVa5cWQMGDNDgwYP13HPPSZI6duyoli1bqmnTpipRokSB/x0XAAAAOLKZWzc0ZMLu3bsVGhoqd3d3NWjQQMYY7d69W3/++ac2bNigunXr5kStuMclJSXJ19dXiYmJ8vHxye1yAEtMmTJFkZGRatu2rYYPH57b5QAAkK7MfF7L0s3qw4YNU9u2bTV79mw5O98cIjk5Wf369dNLL72kb775JivDAgAAALhHZCmI7N692yGESDd/uXvEiBGW3t8AAAAAIH/K0j0iPj4+qX5XQ5KOHz/u8JsXAAAAAJCWLAWRLl26qG/fvlq8eLGOHz+uEydOaNGiRerXr5+6deuW3TUCAAAAKGCydGnWu+++K5vNpp49eyo5OVnGGLm4uOiFF17QxIkTs7tGAAAAAAVMloKIi4uLpk2bpgkTJig+Pl7GGFWqVEkeHh7ZXR8AAACAAihTQaRPnz4Z6jd37twsFQMAAADg3pCpIBIREaEKFSrogQceUBZ+fgQAAAAAJGUyiDz//PNatGiRjhw5oj59+uiZZ55R0aJFc6o2AAAAAAVUpr4168MPP9SpU6f0yiuv6Msvv1RAQIA6d+6s9evXc4YEAAAAQIZl+ut7XV1d1a1bN23cuFEHDx5UjRo1NHDgQFWoUEEXL17MiRoB4J4VHBzs8AgAQEGRpd8RucVms8lms8kYo5SUlOyqCQDw/936NkK+lRAAUNBkOohcvXpVCxcuVIsWLVSlShUdOHBAH3zwgY4dOyYvL6+cqBEAAABAAZOpm9UHDhyoRYsWqXz58nr22We1aNEiFStWLKdqAwAAAFBAZSqIzJw5U+XLl1dQUJCioqIUFRWVZr8VK1ZkS3EAAAAACqZMBZGePXvKZrPlVC0AAAAA7hGZ/kFDAAAAAPi7/ta3ZgEAAABAVhBEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAAAAgOUIIgAAAAAsRxABgDwsKChItWvXVlBQUG6XAgBAtrIZY0xuFwFkRFJSknx9fZWYmCgfH5/cLgcAAAC3ycznNc6IAAAAALAcQQQAAACA5QgiAAAAACxHEAEAAABgOYIIAAAAAMsRRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAgAIqLi5OQ4cOVVxcXG6XkgpBBAAAACigEhIStG/fPiUkJOR2KakQRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAAAAgOUIIgAAAEABdfnyZYfHvIQgAgAAABRQ8fHxDo95CUEEAAAAgOUIIgAAAAAsRxABAAAAYDmCCAAAAADLEUQAAAAAWI4gAgAAAMByBBEAAAAAliOIAAAAALAcQQQAAACA5QgiAAAAACxHEEGaevfurfbt2zu0LVu2TG5ubpo0aZLGjh0rm82m559/3qFPdHS0bDabjh49Kkk6evSobDabSpYsqQsXLjj0rVOnjsaOHZuDRwEAAIC8iiCCDPnkk0/UvXt3ffDBBxoxYoQkyc3NTXPmzNGhQ4fuuv2FCxf07rvv5nSZAAAAyCcIIrirSZMmafDgwVqwYIH69etnb69SpYqaNm2q11577a5jvPjii5oyZYrOnDmTk6UCAAAgnyCIIF0jR47Um2++qdWrV6tjx46p1k+cOFHLly/Xrl270h2nW7duqlSpkt54440M7/vq1atKSkpyWAAAAFAwEERwR2vXrtXbb7+tL774QiEhIWn2qVu3rjp37qyRI0emO5bNZtPEiRP18ccfKz4+PkP7nzBhgnx9fe1LQEBApo8BAAAAeRNBBHdUq1YtBQYGasyYMaluNP+rt956S9u2bdOGDRvSHS80NFSPPvqo/vOf/2Ro/6NGjVJiYqJ9OX78eKbqBwAAQN5FEMEdlS1bVlFRUTp16pRatmx5xzASHBys/v37a+TIkTLGpDvmxIkTtXjxYu3du/eu+3d1dZWPj4/DAgAAgIKBIIJ0lS9fXlFRUTpz5oyeeOKJO96nMWbMGB06dEiLFi1Kd7wGDRqoQ4cOd72UCwAAAAUbQQR3Va5cOW3dulVnz57VE088ocTExFR9SpUqpeHDh2v69Ol3HW/cuHHavHmzYmNjc6JcAAAA5AMEEWTIrcu0zp8/rxYtWuj8+fOp+rz88svy8vK661iVK1dWnz59dOXKlRyoFAAAAPmBc24XgLwpIiIiVZu/v7/++9//3nEbb29v/fbbbw5tgYGBad43MmvWLM2aNetv1wkAAID8iTMiAAAAACxHEAEAAABgOYIIAAAAAMsRRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAABQQAUHBzs85iUEEQAAAKCA8vDwcHjMSwgiAAAAACxHEAEAAABgOYIIAAAAAMsRRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAAAUUEFBQapdu7aCgoJyu5RUbMYYk9tFABmRlJQkX19fJSYmysfHJ7fLAQAAwG0y83mNMyIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAAAAgOUIIgAAAAAsRxABAAAAYDmCCAAAAADLEUQAAAAAWI4gAgAAAMByBBEAAJBvxcXFaejQoYqLi8vtUgBkEkEEAADkWwkJCdq3b58SEhJyuxQAmUQQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAAAAgOUIIgAAAAAsRxABAAAAYDmCCAAAAADLEUQAAAAAWI4gAgAAAMByBBEAAAAAliOIAACAfOvy5csOjwDyD4IIAADIt+Lj4x0eAeQfBBEAAAAAliOIAAAAALAcQQQAAACA5QgiAAAAACxHEAEAAABgOYIIAAAAAMsRRAAAAABYjiACAAAAwHIEEQAAAACWI4gAAAAAsBxBBGk6c+aMnnvuOZUvX16urq4qXbq0QkNDFRUVpeLFi+utt95Kc7sJEyaoePHiunbtmiIiImSz2VStWrVU/ZYsWSKbzabAwMAcPhIAAADkRQQRpKljx47at2+fPv30Ux06dEiRkZF6/PHHdfHiRT3zzDOKiIiQMSbVduHh4erRo4dcXFwkSZ6enjpz5oy+++47h35z585V+fLlLTkWAAAA5D3OuV0A8p7z58/r22+/1datW9WkSRNJUoUKFdSgQQNJUvny5TVt2jR988039vWStG3bNh0+fFh9+/a1tzk7O+vpp5/W3Llz1bBhQ0nSiRMntHXrVg0bNkwLFy608MgAAACQV3BGBKl4eXnJy8tLq1at0tWrV1Otr1mzpurXr6/w8HCH9rlz56pBgwa6//77Hdr79u2rxYsX6/Lly5KkiIgItWzZUqVKlUq3jqtXryopKclhAQAAQMFAEEEqzs7OioiI0Keffio/Pz81atRIo0eP1v79++19+vTpo2XLlunixYuSpIsXL2rp0qUOZ0NuqVOnjoKDg7Vs2TIZYxQREaE+ffrctY4JEybI19fXvgQEBGTfQQIAACBXEUSQpo4dO+rkyZOKjIxUaGiotm7dqrp16yoiIkKS1K1bN6WkpGjx4sWSpMWLF8sYo65du6Y5Xp8+fRQeHq6oqChdvHhRrVq1umsNo0aNUmJion05fvx4th0fAAAAchdBBHfk5uamFi1aaMyYMdqxY4d69+6tsLAwSZKvr686depkvzwrPDxcnTp1ko+PT5pjde/eXd9//73Gjh2rnj17ytn57rcnubq6ysfHx2EBAABAwUAQQYZVr15dly5dsj/v27evtm/frtWrV2v79u1pXpZ1S9GiRdW2bVtFRUVl6LIsAAAAFGwEEaRy9uxZNWvWTJ9//rn279+vhIQELV26VJMmTVK7du3s/Zo0aaJKlSqpZ8+eqlSpkho3bpzuuBEREfr9999VtWrVnD4EAAAA5HF8fS9S8fLy0kMPPaSpU6cqPj5e169fV0BAgPr376/Ro0c79O3Tp49Gjx6tl19++a7juru7y93dPafKBgAAQD5iM2n9Kh2QByUlJcnX11eJiYncLwIAkCRNmTJFkZGRatu2rYYPH57b5QD3vMx8XuPSLAAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAADkW8HBwQ6PAPIPgggAAMi3PDw8HB4B5B8EEQAAAACWI4gAAAAAsBxBBAAAAIDlCCIAAAAALEcQAQAAAGA5gggAAAAAyxFEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAAAAgOUIIgAAAAAsRxABAAD5VlBQkGrXrq2goKDcLgVAJtmMMSa3iwAyIikpSb6+vkpMTJSPj09ulwMAAIDbZObzGmdEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAAAAgOWcc7sAIKNufcFbUlJSLlcCAACAtNz6nJaRL+YliCDfuHDhgiQpICAglysBAABAei5cuCBfX990+/A7Isg3UlJSdPLkSXl7e8tms2X7+ElJSQoICNDx48f5nRKLMOe5g3m3HnNuPeY8dzDv1strc26M0YULF1SmTBk5OaV/FwhnRJBvODk5qVy5cjm+Hx8fnzzxH/K9hDnPHcy79Zhz6zHnuYN5t15emvO7nQm5hZvVAQAAAFiOIAIAAADAcgQR4P9zdXVVWFiYXF1dc7uUewZznjuYd+sx59ZjznMH8269/Dzn3KwOAAAAwHKcEQEAAABgOYIIAAAAAMsRRAAAAABYjiACAAAAwHIEERRYH374oYKCguTm5qYHH3xQ27ZtS7f/1atX9eqrr6pChQpydXVVcHCw5s6d69DnvffeU5UqVeTu7q6AgAANGzZMV65cycnDyHcyM++9e/eWzWZLtdSoUcOh3/Lly1W9enW5urqqevXqWrlyZU4fRr6S3XM+e/ZsPfbYYypSpIiKFCmikJAQ7dy504pDyTdy4u/8lkWLFslms6l9+/Y5VH3+lRPzfv78eQ0aNEj+/v5yc3NTtWrVtGbNmpw+lHwjJ+ac99K7y+xnmPnz56t27dry8PCQv7+/nn32WZ09e9ahT558LzVAAbRo0SJTuHBhM3v2bHPw4EEzdOhQ4+npaX755Zc7btO2bVvz0EMPmY0bN5qEhATzww8/mO3bt9vXf/7558bV1dXMnz/fJCQkmPXr1xt/f3/z0ksvWXFI+UJm5/38+fPm1KlT9uX48eOmaNGiJiwszN5nx44dplChQmb8+PEmJibGjB8/3jg7O5vvv//eoqPK23Jizp9++mkzY8YMs3fvXhMTE2OeffZZ4+vra06cOGHRUeVtOTHntxw9etSULVvWPPbYY6Zdu3Y5eyD5TE7M+9WrV029evVMq1atzLfffmuOHj1qtm3bZqKjoy06qrwtJ+ac99K7y+y8b9u2zTg5OZlp06aZI0eOmG3btpkaNWqY9u3b2/vk1fdSgggKpAYNGpjnn3/eoa1q1apm5MiRafZfu3at8fX1NWfPnr3jmIMGDTLNmjVzaBs+fLh59NFH/37BBURm5/12K1euNDabzRw9etTe1rlzZ9OyZUuHfqGhoaZr165/v+ACICfm/HbJycnG29vbfPrpp3+r1oIip+Y8OTnZNGrUyHzyySemV69eBJHb5MS8f/TRR6ZixYrm2rVr2VprQZETc8576d1ldt7feecdU7FiRYe26dOnm3Llytmf59X3Ui7NQoFz7do17dmzR0888YRD+xNPPKEdO3akuU1kZKTq1aunSZMmqWzZsqpcubL+/e9/688//7T3efTRR7Vnzx77JSpHjhzRmjVr1Lp165w7mHwkK/N+uzlz5igkJEQVKlSwt3333XepxgwNDc3wmAVZTs357S5fvqzr16+raNGif6vegiAn5/yNN95QiRIl1Ldv32yrt6DIqXmPjIxUw4YNNWjQIJUqVUr333+/xo8frxs3bmRr/flRTs0576Xpy8q8P/LIIzpx4oTWrFkjY4x+/fVXLVu2zGFO8+p7qXOu7h3IAb///rtu3LihUqVKObSXKlVKp0+fTnObI0eO6Ntvv5Wbm5tWrlyp33//XQMHDtS5c+fs94l07dpVv/32mx599FEZY5ScnKwXXnhBI0eOzPFjyg+yMu9/derUKa1du1YLFixwaD99+nSWxyzocmrObzdy5EiVLVtWISEhf6vegiCn5nz79u2aM2eOoqOjs7PcAiOn5v3IkSPavHmzunfvrjVr1ujw4cMaNGiQkpOTNWbMmGw9hvwmp+ac99L0ZWXeH3nkEc2fP19dunTRlStXlJycrLZt2+r999+398mr76WcEUGBZbPZHJ4bY1K13ZKSkiKbzab58+erQYMGatWqlaZMmaKIiAj7WZGtW7dq3Lhx+vDDD/Xjjz9qxYoVWr16td58880cP5b8JDPz/lcRERHy8/NL8wbdrI55r8iJOb9l0qRJWrhwoVasWCE3N7e/W2qBkZ1zfuHCBT3zzDOaPXu2ihcvnt2lFijZ/beekpKikiVL6uOPP9aDDz6orl276tVXX9VHH32UnWXna9k957yXZkxm5v3gwYMaMmSIxowZoz179mjdunVKSEjQ888/n+UxrcIZERQ4xYsXV6FChVKl/DNnzqT614Bb/P39VbZsWfn6+trbqlWrJmOMTpw4ofvuu0//+c9/1KNHD/Xr10+SVLNmTV26dEkDBgzQq6++KienezvXZ2XebzHGaO7cuerRo4dcXFwc1pUuXTpLY94LcmrOb3n33Xc1fvx4ff3116pVq1a21Z2f5cScx8fH6+jRo3ryySftbSkpKZIkZ2dnxcbGKjg4OBuPIv/Jqb91f39/FS5cWIUKFbK3VatWTadPn9a1a9fu+N/GvSCn5pz30vRlZd4nTJigRo0a6eWXX5Yk1apVS56ennrsscf01ltvyd/fP8++l97brzYKJBcXFz344IPauHGjQ/vGjRv1yCOPpLlNo0aNdPLkSV28eNHedujQITk5OalcuXKSbl4nf/v/QRYqVEjm5pc+ZPNR5D9ZmfdboqKiFBcXl+a18Q0bNkw15oYNG+465r0gp+Zckt555x29+eabWrdunerVq5dtNed3OTHnVatW1YEDBxQdHW1f2rZtq6ZNmyo6OloBAQHZfhz5TU79rTdq1EhxcXH24Cfd/P9+f3//ezqESDk357yXpi8r836nOZVkn9M8+15q4Y3xgGVuffXdnDlzzMGDB81LL71kPD097d/cMXLkSNOjRw97/wsXLphy5cqZTp06mZ9//tlERUWZ++67z/Tr18/eJywszHh7e5uFCxeaI0eOmA0bNpjg4GDTuXNny48vr8rsvN/yzDPPmIceeijNMbdv324KFSpkJk6caGJiYszEiRPzxFcO5hU5Medvv/22cXFxMcuWLXP4Ks4LFy7k6LHkFzkx57fjW7NSy4l5P3bsmPHy8jKDBw82sbGxZvXq1aZkyZLmrbfeytFjyS9yYs55L727zM57eHi4cXZ2Nh9++KGJj4833377ralXr55p0KCBvU9efS8liKDAmjFjhqlQoYJxcXExdevWNVFRUfZ1vXr1Mk2aNHHoHxMTY0JCQoy7u7spV66cGT58uLl8+bJ9/fXr183YsWNNcHCwcXNzMwEBAWbgwIHmjz/+sOiI8ofMzvv58+eNu7u7+fjjj+845tKlS02VKlVM4cKFTdWqVc3y5ctzqvx8KbvnvEKFCkZSqiWt3724V+XE3/lfEUTSlhPzvmPHDvPQQw8ZV1dXU7FiRTNu3DiTnJycU4eQ72T3nPNemjGZnffp06eb6tWrG3d3d+Pv72+6d++e6ref8uJ7qc0YzoMBAAAAsBb3iAAAAACwHEEEAAAAgOUIIgAAAAAsRxABAAAAYDmCCAAAAADLEUQAAAAAWI4gAgAAAMByBBEAAAAAliOIAAAAALAcQQQAgL/YsWOHChUqpJYtW+Z2KQBQoNmMMSa3iwAAIK/o16+fvLy89Mknn+jgwYMqX758rtRx/fp1FS5cOFf2DQBW4IwIAAD/36VLl7RkyRK98MILatOmjSIiIhzWR0ZGql69enJzc1Px4sXVoUMH+7qrV69qxIgRCggIkKurq+677z7NmTNHkhQRESE/Pz+HsVatWiWbzWZ/PnbsWNWpU0dz585VxYoV5erqKmOM1q1bp0cffVR+fn4qVqyY2rRpo/j4eIexTpw4oa5du6po0aLy9PRUvXr19MMPP+jo0aNycnLS7t27Hfq///77qlChgvi3SAC5iSACAMD/t3jxYlWpUkVVqlTRM888o/DwcPuH9a+++kodOnRQ69attXfvXm3atEn16tWzb9uzZ08tWrRI06dPV0xMjGbOnCkvL69M7T8uLk5LlizR8uXLFR0dLelmOBo+fLh27dqlTZs2ycnJSf/85z+VkpIiSbp48aKaNGmikydPKjIyUvv27dOIESOUkpKiwMBAhYSEKDw83GE/4eHh6t27t0MQAgCrOed2AQAA5BVz5szRM888I0lq2bKlLl68qE2bNikkJETjxo1T165d9frrr9v7165dW5J06NAhLVmyRBs3blRISIgkqWLFipne/7Vr1zRv3jyVKFHC3taxY8dUNZYsWVIHDx7U/fffrwULFui3337Trl27VLRoUUlSpUqV7P379eun559/XlOmTJGrq6v27dun6OhorVixItP1AUB24owIAACSYmNjtXPnTnXt2lWS5OzsrC5dumju3LmSpOjoaDVv3jzNbaOjo1WoUCE1adLkb9VQoUIFhxAiSfHx8Xr66adVsWJF+fj4KCgoSJJ07Ngx+74feOABewi5Xfv27eXs7KyVK1dKkubOnaumTZsqMDDwb9UKAH8XZ0QAANDNMw3JyckqW7asvc0Yo8KFC+uPP/6Qu7v7HbdNb50kOTk5pbof4/r166n6eXp6pmp78sknFRAQoNmzZ6tMmTJKSUnR/fffr2vXrmVo3y4uLurRo4fCw8PVoUMHLViwQO+991662wCAFTgjAgC45yUnJ+uzzz7T5MmTFR0dbV/27dunChUqaP78+apVq5Y2bdqU5vY1a9ZUSkqKoqKi0lxfokQJXbhwQZcuXbK33boHJD1nz55VTEyMXnvtNTVv3lzVqlXTH3/84dCnVq1aio6O1rlz5+44Tr9+/fT111/rww8/1PXr1x1usgeA3MIZEQDAPW/16tX6448/1LdvX/n6+jqs69Spk+bMmaOpU6eqefPmCg4OVteuXZWcnKy1a9dqxIgRCgwMVK9evdSnTx9Nnz5dtWvX1i+//KIzZ86oc+fOeuihh+Th4aHRo0frxRdf1M6dO1N9I1daihQpomLFiunjjz+Wv7+/jh07ppEjRzr06datm8aPH6/27dtrwoQJ8vf31969e1WmTBk1bNhQklStWjU9/PDDeuWVV9SnT5+7nkUBACtwRgQAcM+bM2eOQkJCUoUQ6ebN4tHR0fLx8dHSpUsVGRmpOnXqqFmzZvrhhx/s/T766CN16tRJAwcOVNWqVdW/f3/7GZCiRYvq888/15o1a1SzZk0tXLhQY8eOvWtdTk5OWrRokfbs2aP7779fw4YN0zvvvOPQx8XFRRs2bFDJkiXVqlUr1axZUxMnTlShQoUc+vXt21fXrl1Tnz59sjBDAJD9+EFDAADuAePGjdOiRYt04MCB3C4FACRxRgQAgALt4sWL2rVrl95//30NGTIkt8sBADuCCAAABdjgwYP16KOPqkmTJlyWBSBP4dIsAAAAAJbjjAgAAAAAyxFEAAAAAFiOIAIAAADAcgQRAAAAAJYjiAAAAACwHEEEAAAAgOUIIgAAAAAsRxABAAAAYLn/B5HhT2i5II/XAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,5))\n",
    "sns.boxplot(\n",
    "    x='Accuracy',y='Model',data=results)\n",
    "plt.title('Model Comparison')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab6e30f0-3f15-4495-90ca-d5ef1c6ec169",
   "metadata": {},
   "source": [
    "#### Best Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "334882ab-cd87-4d7b-9127-c540fe3e98fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.84      0.91      0.87        87\n",
      "           1       0.64      0.48      0.55        29\n",
      "\n",
      "    accuracy                           0.80       116\n",
      "   macro avg       0.74      0.70      0.71       116\n",
      "weighted avg       0.79      0.80      0.79       116\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,pred_knn))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d9ba7e9-c63a-43cb-a392-577586bd2827",
   "metadata": {},
   "source": [
    "#### Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "fc7c920b-865e-470a-a1cb-c953532385a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfkAAAHUCAYAAAA5hFEMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA17ElEQVR4nO3de1hVddr/8c8WcAsKFCp7Q6OFio6mKWkRNIaH4IkcladzWmlqWVoTaemQT6nlsJX6aeVZ81RNWU+lY06alEkHtbC0g5mVouYkQxqJIgLB+v3h4552oLJ1w4a13q+udV3xXWt/17248Lq57/VdC5thGIYAAIDpNPJ3AAAAoHaQ5AEAMCmSPAAAJkWSBwDApEjyAACYFEkeAACTIskDAGBSJHkAAEyKJA8AgEmR5NGgfPHFF7rzzjsVExOjJk2aqFmzZrr00kuVlZWln3/+uVbPvXXrViUlJSk8PFw2m01PP/20z89hs9k0adIkn897JkuXLpXNZpPNZtOGDRuq7DcMQ+3atZPNZlOvXr3O6hxz5szR0qVLvfrMhg0bThkTgDML9HcAQE0tXLhQo0aNUocOHfTwww+rU6dOKi8v15YtWzRv3jxt2rRJK1asqLXzDxs2TMXFxVq+fLnOP/98XXTRRT4/x6ZNm/SHP/zB5/PWVGhoqBYtWlQlkefk5GjXrl0KDQ0967nnzJmjFi1aaOjQoTX+zKWXXqpNmzapU6dOZ31ewMpI8mgQNm3apHvvvVfJyclauXKl7Ha7e19ycrLGjh2rtWvX1moMX331le666y6lpqbW2jmuuOKKWpu7Jm6++Wb9/e9/1+zZsxUWFuYeX7RokRISElRUVFQncZSXl8tmsyksLMzv3xOgIaNdjwYhMzNTNptNCxYs8EjwJzVu3FgDBgxwf11ZWamsrCz98Y9/lN1uV2RkpO644w7t37/f43O9evVS586dlZubq549eyokJERt2rTR1KlTVVlZKek/rexff/1Vc+fOdbe1JWnSpEnu//+tk5/Zs2ePe2z9+vXq1auXmjdvruDgYLVu3VrXX3+9jh075j6munb9V199pYEDB+r8889XkyZN1K1bNy1btszjmJNt7ZdfflkTJkxQdHS0wsLCdPXVV2vnzp01+yZLuvXWWyVJL7/8snvs8OHDev311zVs2LBqPzN58mTFx8crIiJCYWFhuvTSS7Vo0SL99m9fXXTRRdq+fbtycnLc37+TnZCTsb/wwgsaO3asLrjgAtntdn3//fdV2vUHDx5Uq1atlJiYqPLycvf8X3/9tZo2barbb7+9xtcKWAFJHvVeRUWF1q9fr+7du6tVq1Y1+sy9996r8ePHKzk5WatWrdITTzyhtWvXKjExUQcPHvQ4Nj8/X4MHD9Ztt92mVatWKTU1VRkZGXrxxRclSf369dOmTZskSTfccIM2bdrk/rqm9uzZo379+qlx48ZavHix1q5dq6lTp6pp06YqKys75ed27typxMREbd++Xc8++6zeeOMNderUSUOHDlVWVlaV4x955BHt3btXzz33nBYsWKDvvvtO/fv3V0VFRY3iDAsL0w033KDFixe7x15++WU1atRIN9988ymvbeTIkXr11Vf1xhtv6LrrrtP999+vJ554wn3MihUr1KZNG8XFxbm/f7+/tZKRkaF9+/Zp3rx5evPNNxUZGVnlXC1atNDy5cuVm5ur8ePHS5KOHTumG2+8Ua1bt9a8efNqdJ2AZRhAPZefn29IMm655ZYaHb9jxw5DkjFq1CiP8Y8//tiQZDzyyCPusaSkJEOS8fHHH3sc26lTJ+O//uu/PMYkGaNHj/YYmzhxolHdP6MlS5YYkoy8vDzDMAzjtddeMyQZ27ZtO23skoyJEye6v77lllsMu91u7Nu3z+O41NRUIyQkxPjll18MwzCM9957z5BkXHvttR7Hvfrqq4YkY9OmTac978l4c3Nz3XN99dVXhmEYxmWXXWYMHTrUMAzDuPjii42kpKRTzlNRUWGUl5cbjz/+uNG8eXOjsrLSve9Unz15vquuuuqU+9577z2P8WnTphmSjBUrVhhDhgwxgoODjS+++OK01whYEZU8TOe9996TpCoLvC6//HJ17NhR7777rse40+nU5Zdf7jF2ySWXaO/evT6LqVu3bmrcuLHuvvtuLVu2TLt3767R59avX6++fftW6WAMHTpUx44dq9JR+O0tC+nEdUjy6lqSkpLUtm1bLV68WF9++aVyc3NP2ao/GePVV1+t8PBwBQQEKCgoSI899pgOHTqkgoKCGp/3+uuvr/GxDz/8sPr166dbb71Vy5Yt08yZM9WlS5cafx6wCpI86r0WLVooJCREeXl5NTr+0KFDkqSoqKgq+6Kjo937T2revHmV4+x2u0pKSs4i2uq1bdtW77zzjiIjIzV69Gi1bdtWbdu21TPPPHPazx06dOiU13Fy/2/9/lpOrl/w5lpsNpvuvPNOvfjii5o3b57at2+vnj17VnvsJ598opSUFEknnn746KOPlJubqwkTJnh93uqu83QxDh06VMePH5fT6eRePHAKJHnUewEBAerbt68+/fTTKgvnqnMy0R04cKDKvh9//FEtWrTwWWxNmjSRJJWWlnqM//6+vyT17NlTb775pg4fPqzNmzcrISFB6enpWr58+Snnb968+SmvQ5JPr+W3hg4dqoMHD2revHm68847T3nc8uXLFRQUpNWrV+umm25SYmKievTocVbnrG4B46kcOHBAo0ePVrdu3XTo0CE99NBDZ3VOwOxI8mgQMjIyZBiG7rrrrmoXqpWXl+vNN9+UJPXp00eS3AvnTsrNzdWOHTvUt29fn8V1coX4F1984TF+MpbqBAQEKD4+XrNnz5YkffbZZ6c8tm/fvlq/fr07qZ/0/PPPKyQkpNYeL7vgggv08MMPq3///hoyZMgpj7PZbAoMDFRAQIB7rKSkRC+88EKVY33VHamoqNCtt94qm82mNWvWyOVyaebMmXrjjTfOeW7AbHhOHg1CQkKC5s6dq1GjRql79+669957dfHFF6u8vFxbt27VggUL1LlzZ/Xv318dOnTQ3XffrZkzZ6pRo0ZKTU3Vnj179Oijj6pVq1Z68MEHfRbXtddeq4iICA0fPlyPP/64AgMDtXTpUv3www8ex82bN0/r169Xv3791Lp1ax0/fty9gv3qq68+5fwTJ07U6tWr1bt3bz322GOKiIjQ3//+d/3zn/9UVlaWwsPDfXYtvzd16tQzHtOvXz9Nnz5dgwYN0t13361Dhw7pqaeeqvYxxy5dumj58uV65ZVX1KZNGzVp0uSs7qNPnDhRH3zwgdatWyen06mxY8cqJydHw4cPV1xcnGJiYryeEzArkjwajLvuukuXX365ZsyYoWnTpik/P19BQUFq3769Bg0apPvuu8997Ny5c9W2bVstWrRIs2fPVnh4uK655hq5XK5q78GfrbCwMK1du1bp6em67bbbdN5552nEiBFKTU3ViBEj3Md169ZN69at08SJE5Wfn69mzZqpc+fOWrVqlfuednU6dOigjRs36pFHHtHo0aNVUlKijh07asmSJV69Oa629OnTR4sXL9a0adPUv39/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RET4ajoAAGodq+t/Jy4uzuObYhiG8vPz9dNPP2nOnDk+De5sfbd+ur9DAGrdL8Xl/g4BqHXO8KBand9n7ex6yuskP3DgQI8k36hRI7Vs2VK9evXSH//4R58GBwAAzp7XSX7SpEm1EAYAAHXP7O16rzsVAQEBKigoqDJ+6NAhBQQE+CQoAADqQiOb77b6yOskbxhGteOlpaVq3LjxOQcEAAB8o8bt+meffVbSidbGc889p2bNmrn3VVRU6P333+eePACgQamvFbiv1DjJz5gxQ9KJSn7evHkerfnGjRvroosu0rx583wfIQAAtcTs9+RrnOTz8vIkSb1799Ybb7yh888/v9aCAgAA587r1fXvvfdebcQBAECdM3u73uuFdzfccIOmTp1aZfzJJ5/UjTfe6JOgAACoCzab77b6yOskn5OTo379+lUZv+aaa/T+++/7JCgAAHDuvG7XHz16tNpH5YKCglRUVOSToAAAqAtm/1OzXlfynTt31iuvvFJlfPny5erUqZNPggIAoC408uFWH3ldyT/66KO6/vrrtWvXLvXp00eS9O677+qll17Sa6+95vMAAQDA2fE6yQ8YMEArV65UZmamXnvtNQUHB6tr165av369wsLCaiNGAABqhcm79WfXYejXr58++ugjFRcX6/vvv9d1112n9PR0de/e3dfxAQBQaxrZbD7bzpbL5ZLNZlN6erp7zDAMTZo0SdHR0QoODlavXr20fft276/vbINav369brvtNkVHR2vWrFm69tprtWXLlrOdDgAAy8nNzdWCBQt0ySWXeIxnZWVp+vTpmjVrlnJzc+V0OpWcnKwjR454Nb9XSX7//v2aMmWK2rRpo1tvvVXnn3++ysvL9frrr2vKlCmKi4vz6uQAAPiTP5+TP3r0qAYPHqyFCxd6vEXWMAw9/fTTmjBhgq677jp17txZy5Yt07Fjx/TSSy95dY4aJ/lrr71WnTp10tdff62ZM2fqxx9/1MyZM706GQAA9Ykv/9RsaWmpioqKPLbS0tJTnnv06NHq16+frr76ao/xvLw85efnKyUlxT1mt9uVlJSkjRs3end9NT1w3bp1GjFihCZPnqx+/frxt+MBAPgNl8ul8PBwj83lclV77PLly/XZZ59Vuz8/P1+S5HA4PMYdDod7X03VOMl/8MEHOnLkiHr06KH4+HjNmjVLP/30k1cn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cMEF2r17twYPHlztvJ06ddILL7ygkpIS9y8Sp4sDQP1Dux6oQ1dffbUSEhKUlpamt99+W3v27NHGjRv1P//zP9qyZYsk6YEHHtDixYu1ePFiffvtt5o4caK2b99+yjmbNGmi8ePHa9y4cXr++ee1a9cubd68WYsWLZIkRUZGKjg4WGvXrtW///1vHT58WNKJF+y4XC4988wz+vbbb/Xll19qyZIlmj59uiRp0KBBatSokYYPH66vv/5ab731lp566qla/g4B8CWSPFCHbDab3nrrLV111VUaNmyY2rdvr1tuuUV79uyRw+GQJN1888167LHHNH78eHXv3l179+7Vvffee9p5H330UY0dO1aPPfaYOnbsqJtvvlkFBQWSpMDAQD377LOaP3++oqOjNXDgQEnSiBEj9Nxzz2np0qXq0qWLkpKStHTpUvcjd82aNdObb76pr7/+WnFxcZowYYKmTZtWi98dAL5mM7jxBgCAKVHJAwBgUiR5AABMiiQPAIBJkeQBADApkjwAACZFkgcAwKRI8gAAmBRJHgAAkyLJAwBgUiR5AABMiiQPAIBJ/X8okOFPzYupCQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cm = confusion_matrix(\n",
    "    y_test,pred_knn\n",
    ")\n",
    "plt.figure(figsize=(6,5))\n",
    "sns.heatmap(\n",
    "    cm,annot=True,fmt='d',cmap='Blues'\n",
    ")\n",
    "plt.title('Confusion Matrix')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13d4119f-a78e-4f5d-82d4-be1c52c99d60",
   "metadata": {},
   "source": [
    "#### Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "b7e7b956-64a9-4ac2-bf71-e4947eb62d14",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Feature</th>\n",
       "      <th>Importance</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Months since First Donation</td>\n",
       "      <td>0.418818</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Months since Last Donation</td>\n",
       "      <td>0.268934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Number of Donations</td>\n",
       "      <td>0.160228</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Total Volume Donated (c.c.)</td>\n",
       "      <td>0.152020</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       Feature  Importance\n",
       "3  Months since First Donation    0.418818\n",
       "0   Months since Last Donation    0.268934\n",
       "1          Number of Donations    0.160228\n",
       "2  Total Volume Donated (c.c.)    0.152020"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "importance = pd.DataFrame({\n",
    "    'Feature':X.columns,\n",
    "    'Importance': rf.feature_importances_\n",
    "})\n",
    "importance.sort_values(\n",
    "    by='Importance',\n",
    "    ascending=False,\n",
    "    inplace=True\n",
    "    )\n",
    "importance"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c0a5cf5e-f448-4fd8-870d-5ed35753ae9b",
   "metadata": {},
   "source": [
    "#### Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "7543e6d9-76bf-4e8d-9be7-37c34e0ccaef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA10AAAHUCAYAAADBW0JmAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABYvklEQVR4nO3deXwNZ///8feRfRchYgmhloglltAShKKx3orWWgS3llZRW7lRS1u1trZSVQRt0Zai7qL2lqg9uJvYgtI2WlvtS8j8/vDL+TqySNKMIK/n4zGPr3PNNTOfuc7c38d595qZWAzDMAQAAAAAMEWu7C4AAAAAAJ5mhC4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESELgAA0ikyMlIWiyXFZcCAAaYcMyYmRiNHjtTJkydN2f8/cfLkSVksFk2cODG7S8m0qKgojRw5Un///Xd2lwLgKWaf3QUAAPCkmTdvngIDA23aChYsaMqxYmJiNGrUKNWpU0cBAQGmHCMni4qK0qhRoxQREaHcuXNndzkAnlKELgAAMqhcuXIKCQnJ7jL+kYSEBFksFtnb58yfAjdu3JCzs3N2lwEgh+D2QgAAstiSJUtUvXp1ubm5yd3dXeHh4dq3b59Nn927d6tt27YKCAiQi4uLAgIC1K5dO/3666/WPpGRkXr55ZclSXXr1rXeyhgZGSlJCggIUERERLLj16lTR3Xq1LF+3rx5sywWixYuXKj+/furUKFCcnJy0rFjxyRJ69evV7169eTp6SlXV1eFhoZqw4YNmTr3pFswN27cqO7du8vHx0eenp7q1KmTrl27pjNnzqh169bKnTu3ChQooAEDBighIcG6fdIti+PHj9f777+vIkWKyNnZWSEhISnWtHXrVtWrV08eHh5ydXVVjRo19N///jfFmn744Qd17dpV+fLlk6urq4YMGaKBAwdKkooVK2Yd382bN0u69z2+8MILKlCggFxcXFSmTBkNHjxY165ds9l/RESE3N3ddezYMTVu3Fju7u7y9/dX//79devWLZu+t27d0ujRo1WmTBk5OzvLx8dHdevWVVRUlLWPYRiaMWOGKlasKBcXF3l7e+ull17S8ePHM/WdAMh+hC4AADLo7t27unPnjs2SZMyYMWrXrp2CgoL01VdfaeHChbpy5Ypq1aqlmJgYa7+TJ0+qdOnSmjx5stauXatx48YpPj5eVatW1blz5yRJTZo00ZgxYyRJH3/8sbZv367t27erSZMmmap7yJAhOnXqlD755BN999138vX11eeff64XXnhBnp6emj9/vr766ivlyZNH4eHhmQ5ekvTvf/9bXl5eWrx4sYYNG6Yvv/xS3bt3V5MmTRQcHKxvvvlGnTt31qRJkzRt2rRk20+fPl1r1qzR5MmT9fnnnytXrlxq1KiRtm/fbu2zZcsWPf/887p06ZLmzJmjRYsWycPDQ82aNdOSJUuS7bNr165ycHDQwoUL9c0336hnz5568803JUnLli2zjm/lypUlSUePHlXjxo01Z84crVmzRn379tVXX32lZs2aJdt3QkKC/vWvf6levXpasWKFunbtqo8++kjjxo2z9rlz544aNWqkd999V02bNtW3336ryMhI1ahRQ6dOnbL2e+2119S3b1/Vr19fy5cv14wZM/TLL7+oRo0a+vPPPzP9nQDIRgYAAEiXefPmGZJSXBISEoxTp04Z9vb2xptvvmmz3ZUrVww/Pz+jdevWqe77zp07xtWrVw03NzdjypQp1vavv/7akGRs2rQp2TZFixY1OnfunKw9LCzMCAsLs37etGmTIcmoXbu2Tb9r164ZefLkMZo1a2bTfvfuXSM4ONioVq1aGqNhGCdOnDAkGRMmTLC2JY3Rg2Pw4osvGpKMDz/80Ka9YsWKRuXKlZPts2DBgsaNGzes7ZcvXzby5Mlj1K9f39r23HPPGb6+vsaVK1esbXfu3DHKlStnFC5c2EhMTLSpqVOnTsnOYcKECYYk48SJE2mea2JiopGQkGBs2bLFkGTs37/fuq5z586GJOOrr76y2aZx48ZG6dKlrZ8XLFhgSDJmz56d6nG2b99uSDImTZpk03769GnDxcXFGDRoUJp1Ang8MdMFAEAGLViwQLt27bJZ7O3ttXbtWt25c0edOnWymQVzdnZWWFiY9bY1Sbp69arefvttlShRQvb29rK3t5e7u7uuXbum2NhYU+pu1aqVzeeoqChduHBBnTt3tqk3MTFRDRs21K5du5LdSpdeTZs2tflcpkwZSUo2S1emTBmbWyqTtGzZ0uaZq6QZrB9//FF3797VtWvXtGPHDr300ktyd3e39rOzs1PHjh3122+/6fDhw2me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Kk2dOlXR0dH69ddftWDBAiUmJqp06dKS7o3zjh07dPLkSZ07dy7Nmcbr16/rzJkz+u2337Rjxw69/fbb6tGjh3r27Gl9m2CHDh3k7Oyszp0763//+582bdqkN998Ux07drQ+z/UwRYoUkaOjo6ZNm6bjx49r5cqVyf72VtGiRWWxWLRq1SqdPXvW5u2ISUqWLKnmzZure/fu2rp1q/bv369XXnlFhQoVUvPmzdM7hAAAAMghHpvQJemht1WNHTtWrVq1UseOHVW5cmUdO3ZMa9eulbe3d7qP0apVKzVs2FB169ZVvnz5tGjRonRt5+joqCFDhqhChQqqXbu27OzstHjx4lTP48cff1Tjxo1VqlQpDRs2TJMmTbK+ACQzli5dqqpVq6pdu3YKCgrSoEGDrLMxFSpU0JYtW3T06FHVqlVLlSpV0vDhw63Ps5khd+7cWrZsmZ5//nmVKVNGn3zyiRYtWqSyZctKkgYMGCA7OzsFBQUpX758aT4zNXv2bBUoUEDPPPOMWrRooZiYGC1ZskQzZsyw9nF1ddXatWt14cIFVa1aVS+99JLq1aun6dOnp7vmfPnyKTIyUl9//bWCgoI0duxYTZw40aZPoUKFNGrUKA0ePFj58+dP9Q2Q8+bNU5UqVdS0aVNVr15dhmHo+++/T3ZLIQAAAGAx0nqPOYBscfnyZXl5eSn4zU9k55T2M30AAACQ9kzo9EiPl/R77dKlSw99Hv+xmukCAAAAgKcNoQsAAAAATEToAgAAAAATEboAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6AAAAAMBEhC4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6AAAAAMBEhC4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAE9lndwEAUvfje+3k6emZ3WUAAADgH2CmCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6AAAAAMBEhC4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwET22V0AgNTVHrZIdk4u2V0GAMBEeyZ0yu4SAJiMmS4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6AAAAAMBEhC4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6AAAAAMBEhC4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESErv/PYrFo+fLlj+RYERERevHFFx/JsZA1HuX1AQAAgKdLtoauiIgIWSwW9ejRI9m6119/XRaLRREREVl6zJEjR6pixYpZus+MmjJliiIjI7O1hsjISOXOndu0/ac3pFgsFuvi5uamkiVLKiIiQnv27DGttrSkdn3Ex8erUaNGj74gAAAAPPGyfabL399fixcv1o0bN6xtN2/e1KJFi1SkSJFsrMw8Xl5epgaeJ828efMUHx+vX375RR9//LGuXr2qZ599VgsWLMju0qz8/Pzk5OSU3WUAAADgCZTtoaty5coqUqSIli1bZm1btmyZ/P39ValSJZu+t27dUu/eveXr6ytnZ2fVrFlTu3btsq7fvHmzLBaLNmzYoJCQELm6uqpGjRo6fPiwpHuzO6NGjdL+/futsyv3zzidO3dOLVq0kKurq0qWLKmVK1da1128eFEdOnRQvnz55OLiopIlS2revHmpntc333yj8uXLy8XFRT4+Pqpfv76uXbsmKfnthXXq1FHv3r01aNAg5cmTR35+fho5cqTN/v7++2+9+uqryp8/v5ydnVWuXDmtWrXKuj4qKkq1a9eWi4uL/P391bt3b+vxMmPNmjWqWbOmcufOLR8fHzVt2lRxcXHW9bdv31avXr1UoEABOTs7KyAgQB988IEkKSAgQJLUokULWSwW6+fU5M6dW35+fgoICNALL7ygb775Rh06dFCvXr108eJFa7+lS5eqbNmycnJyUkBAgCZN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5ODgoC1btqhatWpycnJSgQIFNHjw4DTDSkqzcrlz51ZkZKSk/5vt+eqrr1SrVi25uLioatWqOnLkiHbt2qWQkBC5u7urYcOGOnv2rM1+5s2bpzJlysjZ2VmBgYGaMWPGQ8/TyclJfn5+KlSokCpXrqz//Oc/WrFihVavXm2tSZJOnTql5s2by93dXZ6enmrdurX+/PNP6/qRI0eqYsWKWrhwoQICAuTl5aW2bdva3La3Zs0a1axZU7lz55aPj4+aNm2quLg46/pixYpJkipVqiSLxaI6deqk+9x27typSpUqydnZWSEhIdq3b99Dz/2rr75ScHCwChcubNO+bds2hYWFydXVVd7e3goPD9fFixdT3U9G+//rX//Szp07dfz48YfWCAAAgKfLY/lM17fffqs+ffqof//++t///qfXXntNXbp00aZNmyRJu3btknTvR3l8fLz189WrV9W4cWOtX79e+/btU3h4uJo1a6ZTp06l+9jdunXT119/ratXr1rbtmzZomPHjqlr1676/fff1bhxY1WtWlX79+/XzJkzNWfOHL333nv/+LxHjBihYcOGae/evbK3t1e7du00aNAgTZkyRT/99JPi4uL0zjvvWPvPnj1bQ4cO1fvvv6/Y2FiNGTNGw4cP1/z58zN87Oeff17BwcHW2ycNw9CLL76oCxcuaMuWLVq3bp3i4uLUpk0bm+3i4uK0fPlyrVq1SqtWrdKWLVs0duxY6/pr166pX79+2rVrlzZs2KBcuXKpRYsW1lsZd+7cKUlav3694uPjrcd/2Lldu3ZNTZs2VenSpbVnzx6NHDlSAwYMeOh5/vjjjwoJCbFpi46OVr169VS2bFlt375dW7duVbNmzXT37t0U95HR/pJUtGhR+fr6pjiLKkm3bt3S5cuXbRYAAAA8Heyzu4CUTJw4UREREXr99dclSf369dPPP/+siRMnqm7dusqXL5+ke7NFfn5+1u2Cg4MVHBxs/fzee+/p22+/1cqVK9WrV690Hbt9+/bq37+/vv76a+staHPnzlX16tUVFBSkoUOHyt/fX9OnT5fFYlFgYKD++OMPvf3223rnnXeUK1fmc+yAAQMUHh4uSerTp4/atWunDRs2KDQ0VNK9QHj/TNS7776rSZMmqWXLlpLuzRrFxMRo1qxZ6ty5c4aPHxgYqAMHDki6F4IOHDigEydOyN/fX5K0cOFClS1bVrt27VLVqlUlSYmJiYqMjLQ+H9WxY0dt2LBB77//viSpVatWNseYM2eOfH19FRMTo3Llylm/Sx8fH5vv8mHn9sUXX+ju3buaO3euXF1dVbZsWf3222/q2bNnmud48uRJValSxaZt/PjxCgkJsZlJK1u2bKr7yGj/JIUKFdLJkydTXPfBBx9o1KhRD90HAAAAnjyP5UxXbGysNWgkCQ0NVWxsbJrbXbt2TYMGDVJQUJBy584td3d3HTp0KEMzXblz51bLli2ttxheuXJFS5cuVdeuXa21Va9eXRaLxaa2q1ev6rfffkv3cVJSoUIF67/z588vSSpfvrxN219//SVJOnv2rE6fPq1u3brJ3d3durz33ns2t+9lhGEY1vOKjY2Vv7+/NXBJso7r/d9DQECAzQspChQoYK1RujcT1r59exUvXlyenp7W2wnT+k7Sc26xsbEKDg6Wq6urdbvq1as/9Bxv3LghZ2dnm7akmav0ymj/JC4uLrp+/XqK64YMGaJLly5Zl9OnT2d4/wAAAHg8PZYzXZJsQo1kGwhSM3DgQK1du1YTJ05UiRIl5OLiopdeekm3b9/O0LG7deumevXq6ejRo9qyZYskWW+rS6kOwzBSrPn+c0nqkyQhISFZPwcHB5ttUmpLui0v6f/Onj1bzz77rM1+7OzsHnKGKYuNjbWGotTG+8H2++t7sEZJatasmfz9/TV79mwVLFhQiYmJKleuXJrfSXrO7cHxTK+8efMme/bKxcUlQ/vIaP8kFy5csM7sPcjJyUlOTk6Z2i8AAAAeb4/lTFeZMmW0detWm7aoqCiVKVPG+tnBwSHZMzQ//fSTIiIi1KJFC5UvX15+fn6p3s6Vlrp166p48eKKjIzU3Llz1bp1a+tsTlBQkKKiomx+9EdFRcnDw0OFChVKcX/58uVTfHy89fPRo0dTnfFIr/z586tQoUI6fvy4SpQoYbMkBaeM2Lhxow4ePGi9HTAoKEinTp2ymXGJiYnRpUuXbL6HtJw/f16xsbEaNmyY6tWrpzJlyiQLPI6OjpJk812m59yCgoK0f/9+3bhxw7rdzz///NCaKlWqpJiYGJu2ChUqaMOGDek6p8z0l6SbN28qLi5OlSpVytB2AAAAePI9lqFr4MCBioyM1CeffKKjR4/qww8/1LJly2xelBAQEKANGzbozJkz1h/yJUqU0LJlyxQdHa39+/erffv2mfrbUxaLRV26dNHMmTO1fft2devWzbru9ddf1+nTp/Xmm2/q0KFDWrFihUaMGKF+/fql+jzX888/r+nTp2vv3r3avXu3evTokWyGKDNGjhypDz74QFOmTNGRI0d08OBBzZs3Tx9++GGa2926dUtnzpzR77//rr1792rMmDFq3ry5mjZtqk6dOkmS6tevrwoVKqhDhw7au3evdu7cqU6dOiksLCzZiyhS4+3tLR8fH3366ac6duyYNm7cqH79+tn08fX1lYuLi9asWaM///xTly5dSte5tW/fXrly5VK3bt0UExOj77//XhMnTnxoTeHh4dq+fbtNyBsyZIh27dql119/XQcOHNChQ4c0c+ZMnTt3TpI0ffp0m9sJM9pfuhcInZyc0nULJAAAAJ4uj2XoevHFFzVlyhRNmDBBZcuW1axZszRv3jyb14lPmjRJ69atk7+/v3X24KOPPpK3t7dq1KihZs2aKTw8XJUrV85UDREREbp06ZJKly5t83xZoUKF9P3332vnzp0KDg5Wjx491K1bNw0bNizVfU2aNEn+/v6qXbu22rdvrwEDBtg8i5RZ//73v/XZZ58pMjJS5cuXV1hYmCIjIx8607VmzRoVKFBAAQEBatiwoTZt2qSpU6dqxYoV1tv3kl5z7+3trdq1a6t+/foqXry4lixZku76cuXKpcWLF2vPnj0qV66c3nrrLU2YMMGmj729vaZOnapZs2apYMGCat68ebrOzd3dXd99951iYmJUqVIlDR06VOPGjXtoTY0bN5aDg4PNnxIoVaqUfvjhB+3fv1/VqlVT9erVtWLFCtnb37v79ty5czbPyWW0vyQtWrRIHTp0yJLvHQAAAE8Wi5HZh2OAJ9SMGTO0YsUKrV279pEc7+zZswoMDNTu3bvTfevn5cuX5eXlpeA3P5GdU+aeIQOQsj0TOmV3CQCAp0DS77VLly7J09Mzzb6P7Ys0ALO8+uqrunjxoq5cuWLz5kWznDhxQjNmzMjUs3YAAAB48hG6kOPY29tr6NChj+x41apVU7Vq1R7Z8QAAAPB4eSyf6QIAAACApwWhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6AAAAAMBEhC4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADARoQsAAAAATEToAgAAAAATEboAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6AAAAAMBEhC4AAAAAMBGhCwAAAABMROgCAAAAABMRugAAAADARIQuAAAAADCRfXYXACB1P77XTp6entldBgAAAP4BZroAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6AAAAAMBEhC4AAAAAMBGhCwAAAABMROgCAAAAABPxx5GBx5BhGJKky5cvZ3MlAAAASEnS77Sk321pIXQBj6Hz589Lkvz9/bO5EgAAAKTlypUr8vLySrMPoQt4DOXJk0eSdOrUqYf+jzinuXz5svz9/XX69Gl5enpmdzmPDcYldYxNyhiX1DE2KWNcUsfYpOxpHxfDMHTlyhUVLFjwoX0JXcBjKFeue49benl5PZX/TyoreHp6MjYpYFxSx9ikjHFJHWOTMsYldYxNyp7mcUnvfxznRRoAAAAAYCJCFwAAAACYiNAFPIacnJw0YsQIOTk5ZXcpjx3GJmWMS+oYm5QxLqljbFLGuKSOsUkZ4/J/LEZ63nEIAAAAAMgUZroAAAAAwESELgAAAAAwEaELAAAAAExE6AIAAAAAExG6gEdkxowZKlasmJydnVWlShX99NNPafbfsmWLqlSpImdnZxUvXlyffPJJsj5Lly5VUFCQnJycFBQUpG+//das8k2T1eMSGRkpi8WSbLl586aZp2GKjIxNfHy82rdvr9KlSytXrlzq27dviv1y2jWTnnHJqdfMsmXL1KBBA+XLl0+enp6qXr261q5dm6xfTrtm0jMuOfWa2bp1q0JDQ+Xj4yMXFxcFBgbqo48+StYvp10z6RmXnHrN3G/btm2yt7dXxYoVk617Gq6ZhzIAmG7x4sWGg4ODMXv2bCMmJsbo06eP4ebmZvz6668p9j9+/Ljh6upq9OnTx4iJiTFmz55tODg4GN988421T1RUlGFnZ2eMGTPGiI2NNcaMGWPY29sbP//886M6rX/MjHGZN2+e4enpacTHx9ssT5qMjs2JEyeM3r17G/PnzzcqVqxo9OnTJ1mfnHjNpGdccuo106dPH2PcuHHGzp07jSNHjhhDhgwxHBwcjL1791r75MRrJj3jklOvmb179xpffvml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",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,5))\n",
    "sns.barplot(x='Importance',y='Feature',data=importance)\n",
    "plt.title('Feature Importance')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35f6d0a8-74cf-4a10-83c8-aecbc6615206",
   "metadata": {},
   "source": [
    " #### Model Comparison Report\n",
    "\n",
    "Five machine learning models were developed and evaluated for predicting whether a donor would donate blood in March 2007.\n",
    "\n",
    "The performance of each model was measured using accuracy score.\n",
    "\n",
    "Results:\n",
    "\n",
    "1. KNN achieved the highest accuracy of 80.17%.\n",
    "2. Logistic Regression achieved 75.86%.\n",
    "3. SVM achieved 75.86%.\n",
    "4. Random Forest achieved 71.55%.\n",
    "5. Decision Tree achieved 68.10%.\n",
    "\n",
    "Based on the evaluation results, KNN outperformed all other models and was selected as the final model for prediction."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e94946d3-ce9b-4379-930d-c9220a56510a",
   "metadata": {},
   "source": [
    "#### Observation on KNN\n",
    "KNN performed best because donors with similar donation histories tend to exhibit similar future donation behavior.\n",
    "\n",
    "The algorithm effectively identified patterns among donors based on recency, frequency, and total donation volume."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2d9cfb7-d288-4945-b1f3-aa6336eab611",
   "metadata": {},
   "source": [
    "#### Expected Business Insight"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7dea47a8-697e-46fa-a322-1606b4bf03f2",
   "metadata": {},
   "source": [
    "1. Months since Last Donation\n",
    "   - Donors who donated recently are more likely to donate again.\n",
    "2. Number of Donations\n",
    "   - Regular donors have a higher probability of future donations.\n",
    "3. Total Volume Donated\n",
    "   - Donors with higher cumulative donations are generally more committed and likely to continue donating.\n",
    "4. Months since First Donation\n",
    "   - Long-term donors often show stronger engagement with blood donation programs."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd25d08a-5913-44b2-a76b-7e67d072a3f9",
   "metadata": {},
   "source": [
    "#### Business Impact Report"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f77128f-76e6-42a4-9e6c-42191a132467",
   "metadata": {},
   "source": [
    "\n",
    "\n",
    "1. Understanding donor behavior patterns.\n",
    "\n",
    "2. Identifying factors influencing repeat donations.\n",
    "\n",
    "3. Selecting the most effective classification model.\n",
    "\n",
    "4. Handling possible class imbalance.\n",
    "\n",
    "5. Translating model results into business recommendations."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aba76cd8-7857-42de-9a0a-3cf15835cfa6",
   "metadata": {},
   "source": [
    "#### Challenges faced\n",
    "\n",
    "1. Understanding donor behavior patterns.\n",
    "\n",
    "2. Identifying factors influencing repeat donations.\n",
    "\n",
    "3. Selecting the most effective classification model.\n",
    "\n",
    "4. Handling possible class imbalance.\n",
    "\n",
    "5. Translating model results into business recommendations."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1b3e0453-4b76-4731-a968-4229b0a4eeff",
   "metadata": {},
   "source": [
    "#### Production Recommendation\n",
    "\n",
    "Based on model evaluation results, KNN is recommended for deployment as it achieved the highest prediction accuracy of 80.17%.\n",
    "\n",
    "The blood donation center can use this model to identify potential future donors and target them through personalized awareness campaigns.\n",
    "\n",
    "This approach can improve blood collection efficiency and help maintain adequate blood supplies."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "499c078c-e24f-4074-b689-52a217875552",
   "metadata": {},
   "source": [
    "#### Final Conclusion\n",
    "\n",
    "A predictive model was developed to identify donors likely to donate blood in March 2007.\n",
    "\n",
    "EDA showed that donation frequency and donation recency play a major role in predicting future donations.\n",
    "\n",
    "Multiple machine learning models were evaluated and the best-performing model was selected based on accuracy.\n",
    "\n",
    "The developed model can help blood banks improve donor targeting, optimize campaigns, and maintain a stable blood supply."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5da085da-86e2-43cb-8e4c-a1fc8de90f85",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
