#!/usr/bin/env python3
"""
Check IonCast code for potential issues.
"""

import sys
import os
sys.path.append('/home/LinneaWolniewicz/ionosphere/2025-HL-Ionosphere/scripts/graph_experiment')
sys.path.append('/home/LinneaWolniewicz/ionosphere/graphcast')

def check_imports():
    """Check if all required imports are available."""
    print("Checking imports...")
    
    try:
        import chex
        print("✅ chex imported")
    except ImportError as e:
        print(f"❌ chex import failed: {e}")
    
    try:
        from graphcast import deep_typed_graph_net
        print("✅ deep_typed_graph_net imported")
    except ImportError as e:
        print(f"❌ deep_typed_graph_net import failed: {e}")
    
    try:
        from graphcast import grid_mesh_connectivity
        print("✅ grid_mesh_connectivity imported")
    except ImportError as e:
        print(f"❌ grid_mesh_connectivity import failed: {e}")
    
    try:
        from graphcast import icosahedral_mesh
        print("✅ icosahedral_mesh imported")
    except ImportError as e:
        print(f"❌ icosahedral_mesh import failed: {e}")
    
    try:
        from graphcast import model_utils
        print("✅ model_utils imported")
    except ImportError as e:
        print(f"❌ model_utils import failed: {e}")
    
    try:
        from graphcast import typed_graph
        print("✅ typed_graph imported")
    except ImportError as e:
        print(f"❌ typed_graph import failed: {e}")
    
    try:
        from graphcast import xarray_jax
        print("✅ xarray_jax imported")
    except ImportError as e:
        print(f"❌ xarray_jax import failed: {e}")
    
    try:
        import jax.numpy as jnp
        print("✅ jax.numpy imported")
    except ImportError as e:
        print(f"❌ jax.numpy import failed: {e}")
    
    try:
        import jraph
        print("✅ jraph imported")
    except ImportError as e:
        print(f"❌ jraph import failed: {e}")

def check_graphcast():
    """Check if GraphCast can be imported."""
    print("\nChecking GraphCast...")
    
    try:
        from graphcast.graphcast import GraphCast, ModelConfig, TaskConfig, TASK
        print("✅ GraphCast classes imported successfully")
        
        # Check if task is properly configured
        print(f"Task input variables: {TASK.input_variables}")
        print(f"Task target variables: {TASK.target_variables}")
        
        # Try to create a model config
        config = ModelConfig(
            resolution=1.0,
            mesh_size=3,
            latent_size=32,
            gnn_msg_steps=4,
            hidden_layers=1,
            radius_query_fraction_edge_length=0.6,
        )
        print("✅ ModelConfig created successfully")
        
        # Try to create the model (without initialization)
        model = GraphCast(model_config=config, task_config=TASK)
        print("✅ GraphCast model created successfully")

    except Exception as e:
        print(f"❌ GraphCast import/creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_imports()
    check_graphcast()
    print("\nDone!")
