import unittest

from llm_workers.config import DisplaySettings, UserConfig


class TestDisplaySettings(unittest.TestCase):
    """Test cases for DisplaySettings configuration."""

    def test_default_values(self):
        """Test that DisplaySettings has correct default values."""
        settings = DisplaySettings()
        self.assertTrue(settings.show_token_usage)
        self.assertFalse(settings.show_reasoning)
        self.assertFalse(settings.auto_open_changed_files)
        self.assertTrue(settings.markdown_output)
        # Default is to auto-open image files
        self.assertEqual(settings.file_monitor_include, ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.tiff', '*.svg', '*.wbp'])
        self.assertEqual(settings.file_monitor_exclude, ['.*', '*.log'])

    def test_custom_values(self):
        """Test DisplaySettings with custom values."""
        settings = DisplaySettings(
            show_token_usage=False,
            show_reasoning=True,
            auto_open_changed_files=True,
            markdown_output=True,
            file_monitor_include=['*.py', '*.yaml'],
            file_monitor_exclude=['*.pyc', '*.log', '*.tmp']
        )
        self.assertFalse(settings.show_token_usage)
        self.assertTrue(settings.show_reasoning)
        self.assertTrue(settings.auto_open_changed_files)
        self.assertTrue(settings.markdown_output)
        self.assertEqual(settings.file_monitor_include, ['*.py', '*.yaml'])
        self.assertEqual(settings.file_monitor_exclude, ['*.pyc', '*.log', '*.tmp'])


class TestUserConfigDisplaySettings(unittest.TestCase):
    """Test cases for UserConfig with DisplaySettings integration."""

    def test_default_display_settings(self):
        """Test that UserConfig creates default DisplaySettings."""
        config = UserConfig()
        self.assertIsInstance(config.display_settings, DisplaySettings)
        self.assertTrue(config.display_settings.show_token_usage)
        self.assertFalse(config.display_settings.show_reasoning)

    def test_custom_display_settings(self):
        """Test UserConfig with custom DisplaySettings."""
        custom_settings = DisplaySettings(show_reasoning=True, markdown_output=True)
        config = UserConfig(display_settings=custom_settings)
        self.assertTrue(config.display_settings.show_reasoning)
        self.assertTrue(config.display_settings.markdown_output)

    def test_display_settings_from_dict(self):
        """Test UserConfig creation from dictionary with display_settings."""
        config_data = {
            'models': [],
            'display_settings': {
                'show_token_usage': False,
                'show_reasoning': True,
                'auto_open_changed_files': True,
                'markdown_output': True,
                'file_monitor_include': ['*.py'],
                'file_monitor_exclude': ['*.pyc']
            }
        }
        config = UserConfig(**config_data)
        self.assertFalse(config.display_settings.show_token_usage)
        self.assertTrue(config.display_settings.show_reasoning)
        self.assertTrue(config.display_settings.auto_open_changed_files)
        self.assertTrue(config.display_settings.markdown_output)
        self.assertEqual(config.display_settings.file_monitor_include, ['*.py'])
        self.assertEqual(config.display_settings.file_monitor_exclude, ['*.pyc'])


if __name__ == '__main__':
    unittest.main()
