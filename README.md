# Indoor_Radon_AI_Estimator_Dr.Mutlu-Zeybek
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
