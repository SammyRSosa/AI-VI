"""
tests/test_imports.py — Tests unitarios para el normalizador de transcripciones e importador de YouTube.

Ejecutar con: pytest tests/test_imports.py -v
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Asegura que src/ esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.transcript_parser import (
    clean_transcript_line,
    parse_transcript_text,
    convert_transcript_to_problem,
)
from src.utils.youtube_downloader import (
    extract_youtube_video_id,
    download_youtube_transcript,
    download_and_convert_youtube,
)


class TestTranscriptParser:

    def test_clean_transcript_line_timestamps(self):
        # Probar marcas de tiempo completas
        assert clean_transcript_line("00:00:00.240 Hola a todos") == "Hola a todos"
        assert clean_transcript_line("00:03.120 [Música] Bienvenidos") == "Bienvenidos"
        assert clean_transcript_line(" [12:34] (Aplausos) De acuerdo") == "De acuerdo"
        # Probar marcas en medio de la línea
        assert clean_transcript_line("El tema comenzó a las 10:45 de ayer") == "El tema comenzó a las de ayer"

    def test_clean_transcript_line_brackets(self):
        # Quitar corchetes y paréntesis
        assert clean_transcript_line("[Música] Hola") == "Hola"
        assert clean_transcript_line("(Risas) Jajaja") == "Jajaja"
        assert clean_transcript_line("Hola [otro texto] mundo") == "Hola mundo"

    def test_parse_transcript_text_grouping(self):
        raw_text = """
        # Comentario inicial que debe ignorarse
        00:01 Hola esto es una línea corta.
        00:02 Aquí viene otra línea corta.
        00:03 Y otra más para juntar palabras.
        00:04 Esta tiene bastantes palabras para pasar el límite del párrafo.
        """
        # Agrupar con un target de 10 palabras (muy bajo para forzar cortes rápidos)
        paragraphs = parse_transcript_text(raw_text, target_word_count=10)
        assert len(paragraphs) > 1
        assert "Comentario" not in "".join(paragraphs)
        assert "Hola esto es" in paragraphs[0]

    def test_convert_transcript_to_problem(self):
        raw_text = "00:01 Frase uno. 00:02 Frase dos. 00:03 Frase tres."
        problem = convert_transcript_to_problem(raw_text, "test_inst", target_word_count=4)
        assert problem.name == "test_inst"
        assert problem.n >= 1
        assert problem.elements[0].text is not None


class TestYouTubeDownloader:

    def test_extract_youtube_video_id(self):
        urls = [
            ("https://www.youtube.com/watch?v=JQ4OnXIUwyk", "JQ4OnXIUwyk"),
            ("https://youtu.be/JQ4OnXIUwyk", "JQ4OnXIUwyk"),
            ("https://www.youtube.com/embed/JQ4OnXIUwyk", "JQ4OnXIUwyk"),
            ("https://youtube.com/shorts/JQ4OnXIUwyk?feature=share", "JQ4OnXIUwyk"),
            ("JQ4OnXIUwyk", "JQ4OnXIUwyk"),  # ID directo
        ]
        for url, expected in urls:
            assert extract_youtube_video_id(url) == expected

        assert extract_youtube_video_id("https://google.com") is None
        assert extract_youtube_video_id("invalid_id_long") is None

    @patch("youtube_transcript_api.YouTubeTranscriptApi")
    def test_download_youtube_transcript_success(self, mock_api_class):
        # Mock de la API de YouTube
        mock_api_instance = MagicMock()
        mock_api_class.return_value = mock_api_instance
        mock_api_instance.fetch.return_value = [
            {"text": "Hola", "start": 0.0, "duration": 1.0},
            {"text": "Mundo", "start": 1.0, "duration": 1.0},
        ]
        
        text = download_youtube_transcript("JQ4OnXIUwyk", languages=["es"])
        assert text == "Hola\nMundo"
        mock_api_instance.fetch.assert_called_once_with("JQ4OnXIUwyk", languages=["es"])

    @patch("youtube_transcript_api.YouTubeTranscriptApi")
    def test_download_youtube_transcript_fallback(self, mock_api_class):
        # Mock de la API de YouTube para fallar en primera llamada pero servir traducción en segunda
        mock_api_instance = MagicMock()
        mock_api_class.return_value = mock_api_instance
        
        # Primera llamada a fetch lanza error
        mock_api_instance.fetch.side_effect = Exception("No subtitulos en es")
        
        # Configurar la respuesta de list()
        mock_transcript = MagicMock()
        mock_transcript.is_translatable = True
        mock_translated = MagicMock()
        mock_translated.fetch.return_value = [{"text": "Hello", "start": 0.0}]
        mock_transcript.translate.return_value = mock_translated
        mock_api_instance.list.return_value = [mock_transcript]

        text = download_youtube_transcript("JQ4OnXIUwyk", languages=["es"])
        assert text == "Hello"
        mock_transcript.translate.assert_called_once_with("es")


class TestTwoPhaseBatch:

    @patch("src.llm.client.LLMClient._call_llm_raw")
    def test_ask_best_cut_parsing(self, mock_call):
        from src.llm.client import LLMClient
        from src.problem import SegmentationProblem, TextElement
        
        # Simular respuesta correcta del LLM
        mock_call.return_value = '{"best_cut": 3, "razon": "Corte de prueba exitoso"}'
        
        elements = [TextElement(idx=i, text=f"Texto {i}") for i in range(6)]
        prob = SegmentationProblem(elements=elements, name="test_batch")
        
        client = LLMClient(model="mock-model", verbose=True)
        best_cut, razon = client.ask_best_cut(prob, lo=1, hi=4, window_size=2)
        
        assert best_cut == 3
        assert razon == "Corte de prueba exitoso"
        
    @patch("src.llm.client.LLMClient._call_llm_raw")
    def test_two_phase_solver_batch_mode(self, mock_call):
        from src.solver.two_phase import solve
        from src.llm.client import LLMClient
        from src.problem import SegmentationProblem, TextElement
        
        # Simular respuesta para el mejor corte
        mock_call.return_value = '{"best_cut": 2, "razon": "Corte en index 2"}'
        
        # Crear elementos
        elements = [TextElement(idx=i, text=f"Texto {i}") for i in range(6)]
        prob = SegmentationProblem(elements=elements, name="test_solve_batch", lambda_pen=0.2, min_seg=1)
        
        client = LLMClient(model="mock-model", verbose=True)
        
        seg, stats = solve(prob, client=client, refinement_method="batch")
        
        assert stats["refinement_method"] == "batch"
        # Debería haber hecho a lo sumo 2 llamadas al LLM (una por cada corte ambiguo inicial)
        assert stats["llm_calls"] <= 2
