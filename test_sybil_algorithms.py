#!/usr/bin/env python3
"""
test_sybil_algorithms.py

Comprehensive testing script to compare SybilGuard and SybilLimit algorithms.
Records results and generates comparison reports.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.gdb import GraphDatabase
from sybil.sybilguard.sybilguard import SybilGuard
from sybil.sybillimit.sybillimit import SybilLimit


class SybilAlgorithmTester:
    """Test and compare Sybil detection algorithms"""
    
    def __init__(self, dataset_path: str):
        """Initialize with dataset"""
        self.dataset_path = dataset_path
        self.graph_db = GraphDatabase(dataset_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def test_sybilguard(self, num_seeds: int = 10, method: str = "degree") -> Dict[str, Any]:
        """Test SybilGuard algorithm"""
        print("=" * 60)
        print("TESTING SYBILGUARD")
        print("=" * 60)
        
        start_time = time.time()
        
        # Initialize SybilGuard
        sybilguard = SybilGuard(self.graph_db)
        
        # Run detection
        results = sybilguard.run_detection(num_seeds, method)
        
        # Add timing and algorithm info
        end_time = time.time()
        results['algorithm'] = 'SybilGuard'
        results['execution_time'] = end_time - start_time
        results['parameters'] = {
            'num_seeds': num_seeds,
            'seed_method': method,
            'walk_length': sybilguard.walk_length,
            'num_walks': sybilguard.num_walks,
            'threshold': sybilguard.suspicious_threshold
        }
        
        # Get top suspicious nodes
        top_suspicious = sybilguard.get_top_suspicious_nodes(20)
        results['top_suspicious'] = [
            {
                'node_id': node_id,
                'suspicion_score': score,
                'node_data': self._get_node_info(node_id)
            }
            for node_id, score in top_suspicious
        ]
        
        print(f"SybilGuard completed in {results['execution_time']:.2f} seconds")
        print(f"Detected {len(results['sybil_nodes'])} suspicious nodes")
        
        return results
    
    def test_sybillimit(self, num_seeds: int = 10, method: str = "degree") -> Dict[str, Any]:
        """Test SybilLimit algorithm"""
        print("=" * 60)
        print("TESTING SYBILLIMIT")
        print("=" * 60)
        
        start_time = time.time()
        
        # Initialize SybilLimit
        sybillimit = SybilLimit(self.graph_db)
        
        # Run detection
        results = sybillimit.run_detection(num_seeds, method)
        
        # Add timing and algorithm info
        end_time = time.time()
        results['algorithm'] = 'SybilLimit'
        results['execution_time'] = end_time - start_time
        results['parameters'] = {
            'num_seeds': num_seeds,
            'seed_method': method,
            'walk_length': sybillimit.walk_length,
            'num_walks': sybillimit.num_walks,
            'threshold': sybillimit.acceptance_threshold,
            'mixing_time': sybillimit.mixing_time
        }
        
        # Get top suspicious nodes (lowest acceptance probability)
        top_suspicious = sybillimit.get_most_suspicious_nodes(20)
        results['top_suspicious'] = [
            {
                'node_id': node_id,
                'acceptance_probability': prob,
                'suspicion_score': 1.0 - prob,  # Convert to suspicion for comparison
                'node_data': self._get_node_info(node_id)
            }
            for node_id, prob in top_suspicious
        ]
        
        # Get top trusted nodes
        top_trusted = sybillimit.get_most_trusted_nodes(10)
        results['top_trusted'] = [
            {
                'node_id': node_id,
                'acceptance_probability': prob,
                'node_data': self._get_node_info(node_id)
            }
            for node_id, prob in top_trusted
        ]
        
        print(f"SybilLimit completed in {results['execution_time']:.2f} seconds")
        print(f"Detected {len(results['sybil_nodes'])} suspicious nodes")
        
        return results
    
    def _get_node_info(self, node_id: str) -> Dict[str, Any]:
        """Get human-readable node information"""
        node = self.graph_db.get_node(node_id)
        if not node:
            return {'error': 'Node not found'}
        
        metadata = node.metadata
        return {
            'first_name': metadata.get('first_name', 'Unknown'),
            'last_name': metadata.get('last_name', 'Unknown'),
            'email': metadata.get('email', 'Unknown'),
            'phone': metadata.get('phone', 'Unknown'),
            'degree': len(self.graph_db.get_neighbors(node_id))
        }
    
    def compare_algorithms(self, num_seeds: int = 10, method: str = "degree") -> Dict[str, Any]:
        """Run both algorithms and compare results"""
        print("🔍 COMPREHENSIVE SYBIL ALGORITHM COMPARISON")
        print(f"Dataset: {self.dataset_path}")
        print(f"Total nodes: {len(self.graph_db.nodes)}")
        print(f"Total edges: {len(self.graph_db.edges)}")
        print(f"Seeds: {num_seeds}, Method: {method}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        # Test SybilGuard
        sybilguard_results = self.test_sybilguard(num_seeds, method)
        
        print()
        
        # Test SybilLimit
        sybillimit_results = self.test_sybillimit(num_seeds, method)
        
        print()
        print("=" * 60)
        print("COMPARISON ANALYSIS")
        print("=" * 60)
        
        # Compare results
        comparison = self._analyze_comparison(sybilguard_results, sybillimit_results)
        
        # Package all results
        full_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'dataset_path': self.dataset_path,
                'dataset_stats': {
                    'total_nodes': len(self.graph_db.nodes),
                    'total_edges': len(self.graph_db.edges)
                },
                'test_parameters': {
                    'num_seeds': num_seeds,
                    'seed_method': method
                }
            },
            'sybilguard': sybilguard_results,
            'sybillimit': sybillimit_results,
            'comparison': comparison
        }
        
        self.results = full_results
        return full_results
    
    def _analyze_comparison(self, sg_results: Dict, sl_results: Dict) -> Dict[str, Any]:
        """Analyze and compare the two algorithm results"""
        
        # Basic statistics comparison
        comparison = {
            'execution_times': {
                'sybilguard': sg_results['execution_time'],
                'sybillimit': sl_results['execution_time'],
                'faster_algorithm': 'SybilGuard' if sg_results['execution_time'] < sl_results['execution_time'] else 'SybilLimit'
            },
            'detection_counts': {
                'sybilguard_detected': len(sg_results['sybil_nodes']),
                'sybillimit_detected': len(sl_results['sybil_nodes']),
                'difference': abs(len(sg_results['sybil_nodes']) - len(sl_results['sybil_nodes']))
            },
            'sybil_ratios': {
                'sybilguard': sg_results['sybil_ratio'],
                'sybillimit': sl_results['sybil_ratio'],
                'difference': abs(sg_results['sybil_ratio'] - sl_results['sybil_ratio'])
            }
        }
        
        # Overlap analysis
        sg_sybils = set(sg_results['sybil_nodes'])
        sl_sybils = set(sl_results['sybil_nodes'])
        
        overlap = sg_sybils.intersection(sl_sybils)
        sg_only = sg_sybils - sl_sybils
        sl_only = sl_sybils - sg_sybils
        
        comparison['overlap_analysis'] = {
            'common_detections': len(overlap),
            'sybilguard_only': len(sg_only),
            'sybillimit_only': len(sl_only),
            'agreement_ratio': len(overlap) / max(len(sg_sybils), len(sl_sybils), 1),
            'common_nodes': list(overlap),
            'sybilguard_unique': list(sg_only),
            'sybillimit_unique': list(sl_only)
        }
        
        # Top suspicious nodes comparison
        sg_top_ids = [item['node_id'] for item in sg_results['top_suspicious'][:10]]
        sl_top_ids = [item['node_id'] for item in sl_results['top_suspicious'][:10]]
        
        top_overlap = set(sg_top_ids).intersection(set(sl_top_ids))
        
        comparison['top_suspicious_overlap'] = {
            'common_in_top_10': len(top_overlap),
            'common_nodes': list(top_overlap),
            'overlap_percentage': len(top_overlap) / 10.0 * 100
        }
        
        return comparison
    
    def save_results(self, filename: str = "") -> str:
        """Save results to JSON file"""
        if not filename:
            filename = f"sybil_comparison_{self.timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(self.dataset_path), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Results saved to: {filepath}")
        return filepath
    
    def print_summary_report(self):
        """Print a comprehensive summary report"""
        if not self.results:
            print("No results to report. Run comparison first.")
            return
        
        print("\n" + "=" * 80)
        print("🔍 SYBIL DETECTION ALGORITHM COMPARISON REPORT")
        print("=" * 80)
        
        metadata = self.results['metadata']
        sg = self.results['sybilguard']
        sl = self.results['sybillimit']
        comp = self.results['comparison']
        
        # Dataset info
        print(f"📊 Dataset: {metadata['dataset_path']}")
        print(f"   Nodes: {metadata['dataset_stats']['total_nodes']}")
        print(f"   Edges: {metadata['dataset_stats']['total_edges']}")
        print(f"   Test time: {metadata['timestamp']}")
        print()
        
        # Algorithm performance
        print("⏱️  PERFORMANCE COMPARISON")
        print(f"   SybilGuard execution time: {comp['execution_times']['sybilguard']:.2f}s")
        print(f"   SybilLimit execution time: {comp['execution_times']['sybillimit']:.2f}s")
        print(f"   Faster algorithm: {comp['execution_times']['faster_algorithm']}")
        print()
        
        # Detection results
        print("🎯 DETECTION RESULTS")
        print(f"   SybilGuard detected: {comp['detection_counts']['sybilguard_detected']} nodes ({sg['sybil_ratio']:.1%})")
        print(f"   SybilLimit detected: {comp['detection_counts']['sybillimit_detected']} nodes ({sl['sybil_ratio']:.1%})")
        print(f"   Detection difference: {comp['detection_counts']['difference']} nodes")
        print()
        
        # Agreement analysis
        print("🤝 ALGORITHM AGREEMENT")
        overlap = comp['overlap_analysis']
        print(f"   Common detections: {overlap['common_detections']} nodes")
        print(f"   SybilGuard unique: {overlap['sybilguard_only']} nodes")
        print(f"   SybilLimit unique: {overlap['sybillimit_only']} nodes")
        print(f"   Agreement ratio: {overlap['agreement_ratio']:.1%}")
        print()
        
        # Top suspicious overlap
        print("🔝 TOP SUSPICIOUS NODES OVERLAP")
        top_overlap = comp['top_suspicious_overlap']
        print(f"   Common in top 10: {top_overlap['common_in_top_10']} nodes ({top_overlap['overlap_percentage']:.0f}%)")
        if top_overlap['common_nodes']:
            print(f"   Common nodes: {', '.join(top_overlap['common_nodes'])}")
        print()
        
        # Top suspicious nodes details
        print("🚨 TOP 5 MOST SUSPICIOUS NODES")
        print("\n   SybilGuard Top 5:")
        for i, node in enumerate(sg['top_suspicious'][:5], 1):
            name = f"{node['node_data']['first_name']} {node['node_data']['last_name']}"
            print(f"   {i}. {node['node_id']} ({name}) - Score: {node['suspicion_score']:.3f}")
        
        print("\n   SybilLimit Top 5:")
        for i, node in enumerate(sl['top_suspicious'][:5], 1):
            name = f"{node['node_data']['first_name']} {node['node_data']['last_name']}"
            print(f"   {i}. {node['node_id']} ({name}) - Score: {node['suspicion_score']:.3f}")
        
        if 'top_trusted' in sl:
            print("\n   SybilLimit Most Trusted:")
            for i, node in enumerate(sl['top_trusted'][:3], 1):
                name = f"{node['node_data']['first_name']} {node['node_data']['last_name']}"
                print(f"   {i}. {node['node_id']} ({name}) - Acceptance: {node['acceptance_probability']:.3f}")
        
        print("\n" + "=" * 80)


def main():
    """Command line interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and compare Sybil detection algorithms")
    parser.add_argument("dataset", help="Path to dataset JSON file")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seed nodes (default: 10)")
    parser.add_argument("--method", default="degree", 
                       choices=["degree", "betweenness", "random"], 
                       help="Seed selection method (default: degree)")
    parser.add_argument("--output", "-o", help="Output filename for results (optional)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset file '{args.dataset}' not found")
        return 1
    
    try:
        # Run tests
        tester = SybilAlgorithmTester(args.dataset)
        _ = tester.compare_algorithms(args.seeds, args.method)
        
        # Print summary
        tester.print_summary_report()
        
        # Save results
        if not args.no_save:
            tester.save_results(args.output)
        
        print("\n✅ Testing completed successfully!")
        return 0
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
