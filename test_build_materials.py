#!/usr/bin/env python3
"""Tests for build material PDF routing."""

import os
import tempfile
import unittest
from unittest.mock import patch

import web_chat


class BuildMaterialRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = web_chat.app.test_client()

    def test_build_material_redirects_to_source_when_local_file_missing(self):
        missing_path = "/tmp/does-not-exist/ResidualAGI_WHOLE_BUILD_COMPLETE_2026-08-10-2.pdf"
        with patch.object(web_chat, "BUILD_MATERIAL_LOCAL_PATH", missing_path):
            response = self.client.get("/build-materials/residual-agi")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], web_chat.BUILD_MATERIAL_SOURCE_URL)
        response.close()

    def test_build_material_serves_local_pdf_when_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = "material.pdf"
            local_path = os.path.join(tmp_dir, filename)
            with open(local_path, "wb") as f:
                f.write(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

            with patch.object(web_chat, "BUILD_MATERIAL_DIR", tmp_dir), patch.object(
                web_chat, "BUILD_MATERIAL_FILENAME", filename
            ), patch.object(web_chat, "BUILD_MATERIAL_LOCAL_PATH", local_path):
                response = self.client.get("/build-materials/residual-agi")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "application/pdf")
            self.assertIn(b"%PDF-1.4", response.data)
            response.close()


if __name__ == "__main__":
    unittest.main()
