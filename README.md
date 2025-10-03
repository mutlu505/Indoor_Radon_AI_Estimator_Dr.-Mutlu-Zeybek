# Indoor_Radon_AI_Estimator
Indoor_Radon_AI_Estimator_Dr. Mutlu Zeybek

#1. Python Implementation
class RadonAnalysisSystem:
    def __init__(self):
        self.results = {}
    
    def input_parameters(self, irc, qmax, qmin):
        """Step 1: Input parameters and calculate KQ"""
        self.irc = irc
        self.qmax = qmax
        self.qmin = qmin
        self.kq = qmax / qmin if qmin > 0 else float('inf')
    
    def threshold_analysis(self):
        """Pathway 1: IRC Threshold Analysis"""
        if self.irc > 200:
            return "RISK - Threshold Exceeded"
        elif self.irc < 200:
            return "NO RISK - Below Threshold"
        else:
            return "THRESHOLD VALUE - Critical Point"
    
    def risk_zone_analysis(self):
        """Pathway 2: Radon Risk Zones Analysis"""
        if self.irc <= 100:
            return "LOW RISK ZONE - IRC ≤ 100 Bq/m³"
        elif self.irc <= 300:
            return "MEDIUM RISK ZONE - 100 < IRC ≤ 300 Bq/m³"
        else:
            return "HIGH RISK ZONE - IRC > 300 Bq/m³"
    
    def kq_analysis(self):
        """Pathway 3: Relative Parameter KQ Analysis"""
        if self.kq <= 2:
            return "LOW VARIABILITY - KQ ≤ 2"
        elif self.kq <= 3:
            return "MEDIUM VARIABILITY - 2 < KQ ≤ 3"
        elif self.kq <= 5:
            return "INCREASED VARIABILITY - 3 < KQ ≤ 5"
        elif self.kq <= 10:
            return "HIGH VARIABILITY - 5 < KQ ≤ 10"
        else:
            return "ULTRA HIGH VARIABILITY - KQ > 10"
    
    def consolidate_results(self):
        """Consolidate all results"""
        self.results = {
            'threshold_analysis': self.threshold_analysis(),
            'risk_zone_analysis': self.risk_zone_analysis(),
            'kq_analysis': self.kq_analysis(),
            'irc': self.irc,
            'kq': self.kq
        }
        return self.results
    
    def generate_report(self, take_action=False):
        """Generate comprehensive report"""
        report = f"""
        ===== RADON ANALYSIS COMPREHENSIVE REPORT =====
        
        INPUT PARAMETERS:
        - Indoor Radon Concentration (IRC): {self.irc} Bq/m³
        - Maximum Radon Concentration (Qmax): {self.qmax} Bq/m³
        - Minimum Radon Concentration (Qmin): {self.qmin} Bq/m³
        - Relative Parameter (KQ): {self.kq:.2f}
        
        ANALYSIS RESULTS:
        1. THRESHOLD ANALYSIS: {self.results['threshold_analysis']}
        2. RISK ZONE ANALYSIS: {self.results['risk_zone_analysis']}
        3. VARIABILITY ANALYSIS: {self.results['kq_analysis']}
        
        RECOMMENDATION:"""
        
        if take_action:
            report += "\n        - IMPLEMENT MITIGATION MEASURES (Immediate action required)"
        else:
            report += "\n        - CONTINUE MONITORING (Regular observation recommended)"
        
        report += "\n        =============================================="
        return report
    
    def run_complete_analysis(self, irc, qmax, qmin, take_action=False):
        """Complete analysis workflow"""
        print("Starting Radon Analysis System...")
        
        # Step 1: Input parameters
        self.input_parameters(irc, qmax, qmin)
        print("✓ Parameters loaded and KQ calculated")
        
        # Step 2: Run all analyses
        self.consolidate_results()
        print("✓ All analyses completed")
        
        # Step 3: Generate report
        report = self.generate_report(take_action)
        print("✓ Comprehensive report generated")
        
        return report

# Usage Example
if __name__ == "__main__":
    # Create analysis system
    analyzer = RadonAnalysisSystem()
    
    # Test case 1: High risk scenario
    print("=== TEST CASE 1: HIGH RISK SCENARIO ===")
    report1 = analyzer.run_complete_analysis(
        irc=350,    # High concentration
        qmax=400,   # High variability
        qmin=50,    # Low minimum
        take_action=True
    )
    print(report1)
    
    print("\n" + "="*50 + "\n")
    
    # Test case 2: Low risk scenario
    print("=== TEST CASE 2: LOW RISK SCENARIO ===")
    report2 = analyzer.run_complete_analysis(
        irc=80,     # Low concentration
        qmax=100,   # Low variability
        qmin=80,    # Stable levels
        take_action=False
    )
    print(report2)

    #2. JavaScript Implementation
    class RadonAnalysisSystem {
    constructor() {
        this.results = {};
    }

    // Step 1: Input parameters and calculate KQ
    inputParameters(irc, qmax, qmin) {
        this.irc = irc;
        this.qmax = qmax;
        this.qmin = qmin;
        this.kq = qmin > 0 ? qmax / qmin : Infinity;
    }

    // Pathway 1: IRC Threshold Analysis
    thresholdAnalysis() {
        if (this.irc > 200) {
            return "RISK - Threshold Exceeded";
        } else if (this.irc < 200) {
            return "NO RISK - Below Threshold";
        } else {
            return "THRESHOLD VALUE - Critical Point";
        }
    }

    // Pathway 2: Radon Risk Zones Analysis
    riskZoneAnalysis() {
        if (this.irc <= 100) {
            return "LOW RISK ZONE - IRC ≤ 100 Bq/m³";
        } else if (this.irc <= 300) {
            return "MEDIUM RISK ZONE - 100 < IRC ≤ 300 Bq/m³";
        } else {
            return "HIGH RISK ZONE - IRC > 300 Bq/m³";
        }
    }

    // Pathway 3: Relative Parameter KQ Analysis
    kqAnalysis() {
        if (this.kq <= 2) {
            return "LOW VARIABILITY - KQ ≤ 2";
        } else if (this.kq <= 3) {
            return "MEDIUM VARIABILITY - 2 < KQ ≤ 3";
        } else if (this.kq <= 5) {
            return "INCREASED VARIABILITY - 3 < KQ ≤ 5";
        } else if (this.kq <= 10) {
            return "HIGH VARIABILITY - 5 < KQ ≤ 10";
        } else {
            return "ULTRA HIGH VARIABILITY - KQ > 10";
        }
    }

    // Consolidate all results
    consolidateResults() {
        this.results = {
            thresholdAnalysis: this.thresholdAnalysis(),
            riskZoneAnalysis: this.riskZoneAnalysis(),
            kqAnalysis: this.kqAnalysis(),
            irc: this.irc,
            kq: this.kq
        };
        return this.results;
    }

    // Generate comprehensive report
    generateReport(takeAction = false) {
        return `
===== RADON ANALYSIS COMPREHENSIVE REPORT =====

INPUT PARAMETERS:
- Indoor Radon Concentration (IRC): ${this.irc} Bq/m³
- Maximum Radon Concentration (Qmax): ${this.qmax} Bq/m³
- Minimum Radon Concentration (Qmin): ${this.qmin} Bq/m³
- Relative Parameter (KQ): ${this.kq.toFixed(2)}

ANALYSIS RESULTS:
1. THRESHOLD ANALYSIS: ${this.results.thresholdAnalysis}
2. RISK ZONE ANALYSIS: ${this.results.riskZoneAnalysis}
3. VARIABILITY ANALYSIS: ${this.results.kqAnalysis}

RECOMMENDATION:
- ${takeAction ? 
    "IMPLEMENT MITIGATION MEASURES (Immediate action required)" : 
    "CONTINUE MONITORING (Regular observation recommended)"}
==============================================`;
    }

    // Complete analysis workflow
    runCompleteAnalysis(irc, qmax, qmin, takeAction = false) {
        console.log("Starting Radon Analysis System...");

        // Step 1: Input parameters
        this.inputParameters(irc, qmax, qmin);
        console.log("✓ Parameters loaded and KQ calculated");

        // Step 2: Run all analyses
        this.consolidateResults();
        console.log("✓ All analyses completed");

        // Step 3: Generate report
        const report = this.generateReport(takeAction);
        console.log("✓ Comprehensive report generated");

        return report;
    }
}

// Usage Example
const analyzer = new RadonAnalysisSystem();

// Test case 1: High risk scenario
console.log("=== TEST CASE 1: HIGH RISK SCENARIO ===");
const report1 = analyzer.runCompleteAnalysis(350, 400, 50, true);
console.log(report1);

console.log("\n" + "=".repeat(50) + "\n");

// Test case 2: Low risk scenario
console.log("=== TEST CASE 2: LOW RISK SCENARIO ===");
const report2 = analyzer.runCompleteAnalysis(80, 100, 80, false);
console.log(report2);

#3. Java Implementation
import java.util.HashMap;
import java.util.Map;

public class RadonAnalysisSystem {
    private double irc;
    private double qmax;
    private double qmin;
    private double kq;
    private Map<String, String> results;
    
    public RadonAnalysisSystem() {
        this.results = new HashMap<>();
    }
    
    // Step 1: Input parameters and calculate KQ
    public void inputParameters(double irc, double qmax, double qmin) {
        this.irc = irc;
        this.qmax = qmax;
        this.qmin = qmin;
        this.kq = (qmin > 0) ? qmax / qmin : Double.POSITIVE_INFINITY;
    }
    
    // Pathway 1: IRC Threshold Analysis
    public String thresholdAnalysis() {
        if (this.irc > 200) {
            return "RISK - Threshold Exceeded";
        } else if (this.irc < 200) {
            return "NO RISK - Below Threshold";
        } else {
            return "THRESHOLD VALUE - Critical Point";
        }
    }
    
    // Pathway 2: Radon Risk Zones Analysis
    public String riskZoneAnalysis() {
        if (this.irc <= 100) {
            return "LOW RISK ZONE - IRC ≤ 100 Bq/m³";
        } else if (this.irc <= 300) {
            return "MEDIUM RISK ZONE - 100 < IRC ≤ 300 Bq/m³";
        } else {
            return "HIGH RISK ZONE - IRC > 300 Bq/m³";
        }
    }
    
    // Pathway 3: Relative Parameter KQ Analysis
    public String kqAnalysis() {
        if (this.kq <= 2) {
            return "LOW VARIABILITY - KQ ≤ 2";
        } else if (this.kq <= 3) {
            return "MEDIUM VARIABILITY - 2 < KQ ≤ 3";
        } else if (this.kq <= 5) {
            return "INCREASED VARIABILITY - 3 < KQ ≤ 5";
        } else if (this.kq <= 10) {
            return "HIGH VARIABILITY - 5 < KQ ≤ 10";
        } else {
            return "ULTRA HIGH VARIABILITY - KQ > 10";
        }
    }
    
    // Consolidate all results
    public Map<String, String> consolidateResults() {
        results.put("thresholdAnalysis", thresholdAnalysis());
        results.put("riskZoneAnalysis", riskZoneAnalysis());
        results.put("kqAnalysis", kqAnalysis());
        results.put("irc", String.valueOf(irc));
        results.put("kq", String.format("%.2f", kq));
        return results;
    }
    
    // Generate comprehensive report
    public String generateReport(boolean takeAction) {
        return String.format("""
            ===== RADON ANALYSIS COMPREHENSIVE REPORT =====
            
            INPUT PARAMETERS:
            - Indoor Radon Concentration (IRC): %.1f Bq/m³
            - Maximum Radon Concentration (Qmax): %.1f Bq/m³
            - Minimum Radon Concentration (Qmin): %.1f Bq/m³
            - Relative Parameter (KQ): %.2f
            
            ANALYSIS RESULTS:
            1. THRESHOLD ANALYSIS: %s
            2. RISK ZONE ANALYSIS: %s
            3. VARIABILITY ANALYSIS: %s
            
            RECOMMENDATION:
            - %s
            ===============================================""",
            irc, qmax, qmin, kq,
            results.get("thresholdAnalysis"),
            results.get("riskZoneAnalysis"),
            results.get("kqAnalysis"),
            takeAction ? 
                "IMPLEMENT MITIGATION MEASURES (Immediate action required)" :
                "CONTINUE MONITORING (Regular observation recommended)"
        );
    }
    
    // Complete analysis workflow
    public String runCompleteAnalysis(double irc, double qmax, double qmin, boolean takeAction) {
        System.out.println("Starting Radon Analysis System...");
        
        // Step 1: Input parameters
        inputParameters(irc, qmax, qmin);
        System.out.println("✓ Parameters loaded and KQ calculated");
        
        // Step 2: Run all analyses
        consolidateResults();
        System.out.println("✓ All analyses completed");
        
        // Step 3: Generate report
        String report = generateReport(takeAction);
        System.out.println("✓ Comprehensive report generated");
        
        return report;
    }
    
    // Main method for testing
    public static void main(String[] args) {
        RadonAnalysisSystem analyzer = new RadonAnalysisSystem();
        
        // Test case 1: High risk scenario
        System.out.println("=== TEST CASE 1: HIGH RISK SCENARIO ===");
        String report1 = analyzer.runCompleteAnalysis(350, 400, 50, true);
        System.out.println(report1);
        
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        // Test case 2: Low risk scenario
        System.out.println("=== TEST CASE 2: LOW RISK SCENARIO ===");
        String report2 = analyzer.runCompleteAnalysis(80, 100, 80, false);
        System.out.println(report2);
    }
}


#complete Python implementation of the GIRA framework:
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PhysicsInformedRadonModel:
    """
    Geologically-Informed Radon Assessment (GIRA) Framework
    Physics-Informed Neural Network for Indoor Radon Concentration Prediction
    """
    
    def __init__(self):
        self.porosity_data = {
            'Volcanic Rocks': 0.10,
            'Intrusive igneous and metamorphic rocks': 0.025,
            'Sedimentary rocks': 0.14,
            'Concrete': 0.14,
            'Cement': 0.058
        }
        
    def calculate_porosity(self, building_materials):
        """Calculate effective porosity based on building materials"""
        if isinstance(building_materials, list):
            return np.mean([self.porosity_data.get(mat, 0.1) for mat in building_materials])
        else:
            return self.porosity_data.get(building_materials, 0.1)

class PINN(nn.Module):
    """Physics-Informed Neural Network for Radon Concentration Prediction"""
    
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=1):
        super(PINN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class RadonPredictor:
    """Main class for radon concentration prediction and risk classification"""
    
    def __init__(self):
        self.pinn_model = None
        self.ann_model = None
        self.svr_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.physics_model = PhysicsInformedRadonModel()
        
    def prepare_features(self, df):
        """Prepare features for model training"""
        features = []
        
        for idx, row in df.iterrows():
            # Basic geographical features
            northing = row['Northing (X)']
            easting = row['Easting (Y)']
            elevation = row['Elevation (Z) (meters)']
            
            # Soil radon concentration
            soil_radon = row['Mean soil radon concentration (Qg+Qf) (Bq/m3)']
            
            # Building material porosity
            building_materials = row['Building material types of building']
            porosity = self.physics_model.calculate_porosity(building_materials)
            
            # Rock type encoding
            rock_type = row['Basement Rock']
            rock_encoding = self._encode_rock_type(rock_type)
            
            feature_vector = [
                northing, easting, elevation, 
                soil_radon, porosity, rock_encoding
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def _encode_rock_type(self, rock_type):
        """Encode rock types numerically"""
        rock_encoding = {
            'Volcanic Rock': 1.0,
            'Limestone': 0.7,
            'Serpentinite': 0.5,
            'Schist': 0.3
        }
        return rock_encoding.get(rock_type, 0.5)
    
    def physics_loss(self, predictions, features, measured_qt=None):
        """
        Physics-based loss function incorporating the hypotheses from the paper
        """
        loss = 0.0
        
        # Extract features
        soil_radon = features[:, 3]  # Qg + Qf
        porosity = features[:, 4]    # n
        rock_encoding = features[:, 5]  # Rock type
        
        # Hypothesis 7 & 8: Porosity constraints
        # When n=0, Qt should be close to Qb (building contribution)
        # When n=1, Qt should be close to 0
        building_contribution = 100  # Base Qb estimate
        
        # Physics constraint 1: High porosity should limit radon accumulation
        porosity_constraint = torch.abs(predictions * porosity)
        loss += 0.1 * torch.mean(porosity_constraint)
        
        # Physics constraint 2: Soil radon should correlate with predictions
        if measured_qt is not None:
            soil_correlation = torch.abs(predictions.squeeze() - 0.1 * torch.tensor(soil_radon, dtype=torch.float32))
            loss += 0.05 * torch.mean(soil_correlation)
        
        # Physics constraint 3: Predictions should be positive
        positivity_constraint = torch.relu(-predictions)
        loss += 0.1 * torch.mean(positivity_constraint)
        
        return loss
    
    def train_pinn(self, X, y, epochs=1000, lr=0.001, physics_weight=0.5):
        """Train the Physics-Informed Neural Network"""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        self.pinn_model = PINN(input_dim=X.shape[1])
        optimizer = optim.Adam(self.pinn_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Data prediction
            predictions = self.pinn_model(X_tensor)
            data_loss = nn.MSELoss()(predictions, y_tensor)
            
            # Physics loss
            physics_loss = self.physics_loss(predictions, X_tensor, y)
            
            # Total loss
            total_loss = data_loss + physics_weight * physics_loss
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 200 == 0:
                print(f'Epoch {epoch}, Total Loss: {total_loss.item():.4f}, '
                      f'Data Loss: {data_loss.item():.4f}, '
                      f'Physics Loss: {physics_loss.item():.4f}')
    
    def train_benchmark_models(self, X, y):
        """Train traditional machine learning models for benchmarking"""
        # Standard ANN
        self.ann_model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        optimizer = optim.Adam(self.ann_model.parameters(), lr=0.001)
        for epoch in range(500):
            optimizer.zero_grad()
            predictions = self.ann_model(X_tensor)
            loss = nn.MSELoss()(predictions, y_tensor)
            loss.backward()
            optimizer.step()
        
        # SVR and Random Forest
        self.svr_model = SVR(kernel='rbf', C=1.0)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.svr_model.fit(X, y)
        self.rf_model.fit(X, y)
    
    def predict(self, X, model_type='pinn'):
        """Make predictions using specified model"""
        if model_type == 'pinn' and self.pinn_model:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                return self.pinn_model(X_tensor).numpy().flatten()
        elif model_type == 'ann' and self.ann_model:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                return self.ann_model(X_tensor).numpy().flatten()
        elif model_type == 'svr' and self.svr_model:
            return self.svr_model.predict(X)
        elif model_type == 'rf' and self.rf_model:
            return self.rf_model.predict(X)
        else:
            raise ValueError("Model not trained or invalid model type")
    
    def risk_classification(self, qt_values):
        """Classify radon risk based on concentration levels"""
        classifications = []
        zones = []
        
        for qt in qt_values:
            # Threshold-based classification
            if qt > 200:
                risk_class = "RISK"
            elif qt < 200:
                risk_class = "NO RISK"
            else:
                risk_class = "THRESHOLD (CRITICAL) VALUE"
            
            # Risk zone classification
            if qt <= 100:
                risk_zone = "LOW RISK ZONE"
            elif 100 < qt <= 300:
                risk_zone = "MEDIUM RISK ZONE"
            else:
                risk_zone = "HIGH RISK ZONE"
            
            classifications.append(risk_class)
            zones.append(risk_zone)
        
        return classifications, zones
    
    def calculate_kq_parameter(self, q_max, q_min):
        """Calculate Relative Parameter of Radon Activity (KQ)"""
        kq = q_max / q_min if q_min > 0 else 0
        
        if kq <= 2:
            return kq, "LOW"
        elif 2 < kq <= 3:
            return kq, "MEDIUM"
        elif 3 < kq <= 5:
            return kq, "INCREASED"
        elif 5 < kq <= 10:
            return kq, "HIGH"
        else:
            return kq, "ULTRA HIGH"

def create_sample_dataset():
    """Create sample dataset based on the paper's Table 1"""
    data = {
        'Code Name': ['M60', 'M59', 'M63', 'M50', 'M61', 'M53', 'M45', 
                     'M64', 'M74', 'M48', 'M84', 'M47', 'M80', 'M58'],
        'Northing (X)': [622455, 622869, 619616, 621217, 618030, 615007, 
                        636472, 712200, 620914, 625899, 579297, 625933, 
                        589540, 624625],
        'Easting (Y)': [4114254, 4114699, 4101966, 4116305, 4102360, 4121899,
                       4110810, 4055036, 4119927, 4107435, 4098688, 4107357,
                       4099627, 4114379],
        'Elevation (Z) (meters)': [650, 625, 24, 695, 48, 647, 715, 121, 
                                  680, 611, 6, 611, 400, 638],
        'Indoor radon concentration of building (Qt) (Bq/m3)': [657, 753, 529, 1180, 625, 1172, 
                                                               2809, 453, 383, 1638, 324, 1660, 
                                                               357, 954],
        'Building material types of building': ['Concrete', 'Concrete', 'Concrete, Volcanic Rock', 
                                               'Concrete', 'Concrete, Red Tiles', 'Volcanic Rock',
                                               'Volcanic Rock', 'Volcanic Rock', 'Concrete', 
                                               'Cement, Limestone', 'Concrete, Iron', 'Volcanic Rock',
                                               'Concrete, Limestone, Red Tiles', 'Concrete'],
        'Mean soil radon concentration (Qg+Qf) (Bq/m3)': [94600, 69000, 49400, 42600, 27600, 26400, 
                                                         15500, 12500, 11600, 11000, 10400, 6200, 
                                                         6000, 5830],
        'Basement Rock': ['Volcanic Rock', 'Volcanic Rock', 'Volcanic Rock', 'Volcanic Rock', 
                         'Limestone', 'Volcanic Rock', 'Serpentinite', 'Limestone', 'Limestone',
                         'Limestone', 'Limestone', 'Limestone', 'Schist', 'Serpentinite']
    }
    
    return pd.DataFrame(data)

def main():
    """Main demonstration of the GIRA framework"""
    print("=== GIRA Framework: Physics-Informed AI for Radon Assessment ===\n")
    
    # Create sample dataset
    df = create_sample_dataset()
    print("Sample Dataset:")
    print(df[['Code Name', 'Indoor radon concentration of building (Qt) (Bq/m3)', 
              'Basement Rock']].to_string(index=False))
    print("\n" + "="*50 + "\n")
    
    # Initialize predictor
    predictor = RadonPredictor()
    
    # Prepare features and target
    X = predictor.prepare_features(df)
    y = df['Indoor radon concentration of building (Qt) (Bq/m3)'].values
    
    # Normalize features
    X_scaled = predictor.scaler.fit_transform(X)
    
    # Train models
    print("Training Physics-Informed Neural Network...")
    predictor.train_pinn(X_scaled, y, epochs=1000)
    
    print("\nTraining Benchmark Models...")
    predictor.train_benchmark_models(X_scaled, y)
    
    # Make predictions
    pinn_predictions = predictor.predict(X_scaled, 'pinn')
    ann_predictions = predictor.predict(X_scaled, 'ann')
    svr_predictions = predictor.predict(X_scaled, 'svr')
    rf_predictions = predictor.predict(X_scaled, 'rf')
    
    # Calculate performance metrics
    def calculate_metrics(true, pred):
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        r2 = 1 - np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2)
        return mae, rmse, r2
    
    pinn_mae, pinn_rmse, pinn_r2 = calculate_metrics(y, pinn_predictions)
    ann_mae, ann_rmse, ann_r2 = calculate_metrics(y, ann_predictions)
    svr_mae, svr_rmse, svr_r2 = calculate_metrics(y, svr_predictions)
    rf_mae, rf_rmse, rf_r2 = calculate_metrics(y, rf_predictions)
    
    # Display results
    print("\n" + "="*80)
    print("COMPARATIVE PERFORMANCE OF PREDICTIVE MODELS")
    print("="*80)
    print(f"{'Model':<25} {'MAE (Bq/m³)':<15} {'RMSE (Bq/m³)':<15} {'R²':<10}")
    print("-"*80)
    print(f"{'Support Vector Regression':<25} {svr_mae:<15.2f} {svr_rmse:<15.2f} {svr_r2:<10.4f}")
    print(f"{'Random Forest':<25} {rf_mae:<15.2f} {rf_rmse:<15.2f} {rf_r2:<10.4f}")
    print(f"{'Standard ANN':<25} {ann_mae:<15.2f} {ann_rmse:<15.2f} {ann_r2:<10.4f}")
    print(f"{'Proposed PINN (This Study)':<25} {pinn_mae:<15.2f} {pinn_rmse:<15.2f} {pinn_r2:<10.4f}")
    
    # Risk classification
    risk_classes, risk_zones = predictor.risk_classification(pinn_predictions)
    
    print("\n" + "="*80)
    print("AI-BASED RADON RISK CLASSIFICATION RESULTS")
    print("="*80)
    print(f"{'Code':<6} {'Measured Qt':<12} {'Predicted Qt':<12} {'Error %':<10} {'Risk Class':<20} {'Risk Zone':<20}")
    print("-"*80)
    
    for i, code in enumerate(df['Code Name']):
        measured = df['Indoor radon concentration of building (Qt) (Bq/m3)'].iloc[i]
        predicted = pinn_predictions[i]
        error_pct = np.abs(measured - predicted) / measured * 100
        
        print(f"{code:<6} {measured:<12} {predicted:<12.1f} {error_pct:<10.1f} "
              f"{risk_classes[i]:<20} {risk_zones[i]:<20}")
    
    # Demonstrate KQ parameter calculation
    print("\n" + "="*50)
    print("RELATIVE PARAMETER OF RADON ACTIVITY (KQ)")
    print("="*50)
    
    q_max = np.max(y)
    q_min = np.min(y)
    kq_value, kq_class = predictor.calculate_kq_parameter(q_max, q_min)
    
    print(f"Maximum Radon Concentration (Qmax): {q_max} Bq/m³")
    print(f"Minimum Radon Concentration (Qmin): {q_min} Bq/m³")
    print(f"KQ Parameter: {kq_value:.2f}")
    print(f"Radon Activity Classification: {kq_class}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    models = ['SVR', 'RF', 'ANN', 'PINN']
    mae_scores = [svr_mae, rf_mae, ann_mae, pinn_mae]
    plt.bar(models, mae_scores, color=['red', 'orange', 'blue', 'green'])
    plt.title('Model Performance (MAE)')
    plt.ylabel('MAE (Bq/m³)')
    
    plt.subplot(1, 3, 2)
    # Actual vs Predicted plot
    plt.scatter(y, pinn_predictions, alpha=0.7, color='green', label='PINN Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Measured Qt (Bq/m³)')
    plt.ylabel('Predicted Qt (Bq/m³)')
    plt.title('PINN: Actual vs Predicted')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Risk zone distribution
    zone_counts = {'LOW': risk_zones.count('LOW RISK ZONE'),
                  'MEDIUM': risk_zones.count('MEDIUM RISK ZONE'),
                  'HIGH': risk_zones.count('HIGH RISK ZONE')}
    plt.pie(zone_counts.values(), labels=zone_counts.keys(), autopct='%1.1f%%', 
            colors=['green', 'orange', 'red'])
    plt.title('Risk Zone Distribution')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
