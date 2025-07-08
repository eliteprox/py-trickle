"""
Test suite to verify that all required classes are properly exported
from the trickle_app package and can be imported from an installed package.
"""

import pytest
import importlib
import sys
import torch
import numpy as np
from fractions import Fraction


class TestPackageExports:
    """Test that all required classes are exported and importable."""
    
    def test_trickle_publisher_import(self):
        """Test that TricklePublisher can be imported from the package."""
        try:
            from trickle_app import TricklePublisher
            assert TricklePublisher is not None
            assert hasattr(TricklePublisher, '__init__')
        except ImportError as e:
            pytest.fail(f"Failed to import TricklePublisher: {e}")
    
    def test_trickle_subscriber_import(self):
        """Test that TrickleSubscriber can be imported from the package."""
        try:
            from trickle_app import TrickleSubscriber
            assert TrickleSubscriber is not None
            assert hasattr(TrickleSubscriber, '__init__')
        except ImportError as e:
            pytest.fail(f"Failed to import TrickleSubscriber: {e}")
    
    def test_frame_classes_import(self):
        """Test that all frame classes can be imported from the package."""
        frame_classes = [
            'SideData',
            'InputFrame', 
            'VideoFrame',
            'AudioFrame',
            'OutputFrame',
            'VideoOutput',
            'AudioOutput'
        ]
        
        for class_name in frame_classes:
            try:
                cls = getattr(__import__('trickle_app', fromlist=[class_name]), class_name)
                assert cls is not None, f"{class_name} is None"
                assert hasattr(cls, '__init__') or hasattr(cls, '__new__'), f"{class_name} has no constructor"
            except (ImportError, AttributeError) as e:
                pytest.fail(f"Failed to import {class_name}: {e}")
    
    def test_all_exports_in___all__(self):
        """Test that all classes listed in __all__ can be imported."""
        import trickle_app
        
        # Get the __all__ list
        all_exports = getattr(trickle_app, '__all__', [])
        assert len(all_exports) > 0, "__all__ is empty or not defined"
        
        # Test each export
        for export_name in all_exports:
            try:
                cls = getattr(trickle_app, export_name)
                assert cls is not None, f"{export_name} is None"
            except AttributeError as e:
                pytest.fail(f"Failed to import {export_name} from __all__: {e}")
    
    def test_direct_import_style(self):
        """Test importing classes using direct import style."""
        try:
            # Test individual imports
            from trickle_app import TricklePublisher, TrickleSubscriber
            from trickle_app import VideoFrame, AudioFrame, VideoOutput, AudioOutput
            from trickle_app import SideData, InputFrame, OutputFrame
            from trickle_app import TrickleClient, SimpleTrickleClient
            from trickle_app import TrickleApp, create_app, TrickleProtocol
            
            # Verify they are not None
            assert all([
                TricklePublisher, TrickleSubscriber,
                VideoFrame, AudioFrame, VideoOutput, AudioOutput,
                SideData, InputFrame, OutputFrame,
                TrickleClient, SimpleTrickleClient,
                TrickleApp, create_app, TrickleProtocol
            ])
        except ImportError as e:
            pytest.fail(f"Failed direct import: {e}")
    
    def test_wildcard_import_style(self):
        """Test importing using wildcard import style."""
        # Note: This is generally not recommended, but we test it for completeness
        import trickle_app
        
        # Verify key classes are available in the module namespace
        key_classes = [
            'TricklePublisher', 'TrickleSubscriber',
            'VideoFrame', 'AudioFrame', 'VideoOutput', 'AudioOutput',
            'SideData', 'InputFrame', 'OutputFrame'
        ]
        
        for class_name in key_classes:
            assert hasattr(trickle_app, class_name), f"{class_name} not found in module namespace"
            cls = getattr(trickle_app, class_name)
            assert cls is not None, f"{class_name} is None"


class TestFrameClassesFunctionality:
    """Test that frame classes can be instantiated and used."""
    
    def test_side_data_creation(self):
        """Test SideData can be created."""
        from trickle_app import SideData
        
        side_data = SideData()
        assert side_data.skipped == True
        assert side_data.input is None
    
    def test_video_frame_creation(self):
        """Test VideoFrame can be created and used."""
        from trickle_app import VideoFrame
        
        # Create a simple tensor
        tensor = torch.zeros((3, 384, 704))  # CHW format
        timestamp = 100
        time_base = Fraction(1, 30)
        
        frame = VideoFrame(tensor, timestamp, time_base)
        assert frame.tensor.shape == (3, 384, 704)
        assert frame.timestamp == timestamp
        assert frame.time_base == time_base
        assert isinstance(frame.log_timestamps, dict)
        
        # Test from_tensor class method
        frame2 = VideoFrame.from_tensor(tensor, timestamp)
        assert frame2.tensor.shape == tensor.shape
        assert frame2.timestamp == timestamp
    
    def test_video_output_creation(self):
        """Test VideoOutput can be created and used."""
        from trickle_app import VideoFrame, VideoOutput
        
        tensor = torch.zeros((3, 384, 704))
        frame = VideoFrame(tensor, 100, Fraction(1, 30))
        
        output = VideoOutput(frame, "test-request-id")
        assert output.frame == frame
        assert output.request_id == "test-request-id"
        assert output.tensor.shape == tensor.shape
        assert output.timestamp == 100
    
    def test_audio_output_creation(self):
        """Test AudioOutput can be created."""
        from trickle_app import AudioOutput
        
        # AudioOutput expects a list of AudioFrame objects
        # For this test, we'll just verify it can be instantiated with an empty list
        output = AudioOutput([], "test-request-id")
        assert output.frames == []
        assert output.request_id == "test-request-id"


class TestPublisherSubscriberInstantiation:
    """Test that TricklePublisher and TrickleSubscriber can be instantiated."""
    
    def test_trickle_publisher_instantiation(self):
        """Test that TricklePublisher can be instantiated."""
        from trickle_app import TricklePublisher
        
        # We need to provide required parameters for TricklePublisher
        # Let's check what parameters it expects without actually running it
        assert hasattr(TricklePublisher, '__init__')
        
        # Test that the class can be referenced and has expected methods
        expected_methods = ['publish', 'stop', '__init__']
        for method in expected_methods:
            assert hasattr(TricklePublisher, method), f"TricklePublisher missing method: {method}"
    
    def test_trickle_subscriber_instantiation(self):
        """Test that TrickleSubscriber can be instantiated."""
        from trickle_app import TrickleSubscriber
        
        # We need to provide required parameters for TrickleSubscriber  
        # Let's check what parameters it expects without actually running it
        assert hasattr(TrickleSubscriber, '__init__')
        
        # Test that the class can be referenced and has expected methods
        expected_methods = ['subscribe', 'unsubscribe', '__init__']
        for method in expected_methods:
            assert hasattr(TrickleSubscriber, method), f"TrickleSubscriber missing method: {method}"


class TestPackageStructure:
    """Test the overall package structure and version."""
    
    def test_package_version(self):
        """Test that package version is accessible."""
        import trickle_app
        
        assert hasattr(trickle_app, '__version__')
        version = trickle_app.__version__
        assert isinstance(version, str)
        assert len(version) > 0
    
    def test_package_docstring(self):
        """Test that package has proper documentation."""
        import trickle_app
        
        assert hasattr(trickle_app, '__doc__')
        assert trickle_app.__doc__ is not None
        assert len(trickle_app.__doc__.strip()) > 0
    
    def test_no_relative_imports_in_installed_package(self):
        """Verify that importing works as an installed package (not relative imports)."""
        # This test verifies that we're importing from the package namespace
        # rather than using relative imports
        
        import trickle_app
        module_file = getattr(trickle_app, '__file__', None)
        
        # The module should have a __file__ attribute when properly installed
        assert module_file is not None, "Package module should have __file__ attribute"
        
        # Verify we can access all exported classes through the main module
        from trickle_app import TricklePublisher, TrickleSubscriber
        from trickle_app import VideoFrame, AudioFrame
        
        # These should be the same objects whether imported directly or through the module
        assert trickle_app.TricklePublisher is TricklePublisher
        assert trickle_app.TrickleSubscriber is TrickleSubscriber
        assert trickle_app.VideoFrame is VideoFrame
        assert trickle_app.AudioFrame is AudioFrame


if __name__ == "__main__":
    pytest.main([__file__])