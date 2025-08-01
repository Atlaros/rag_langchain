#!/usr/bin/env python3
"""
Script de pruebas para el microservicio RAG
Permite probar todos los endpoints principales del microservicio
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, Any

# Configuración del microservicio
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

class RAGMicroserviceClient:
    """Cliente para interactuar con el microservicio RAG"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = TIMEOUT
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica que el servicio esté funcionando"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado completo del servicio"""
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_namespaces(self) -> Dict[str, Any]:
        """Lista todos los namespaces disponibles"""
        try:
            response = self.session.get(f"{self.base_url}/namespaces")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def upload_document(self, pdf_path: str, namespace: str, description: str = None) -> Dict[str, Any]:
        """Sube un documento PDF al microservicio"""
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
                data = {'namespace': namespace}
                if description:
                    data['description'] = description
                
                response = self.session.post(
                    f"{self.base_url}/documents/upload",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def query(self, query: str, namespace: str = "compromidos", **kwargs) -> Dict[str, Any]:
        """Realiza una consulta RAG"""
        try:
            payload = {
                "query": query,
                "namespace": namespace,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def batch_query(self, queries: list, namespace: str = "compromidos", **kwargs) -> Dict[str, Any]:
        """Realiza múltiples consultas en lote"""
        try:
            payload = {
                "queries": queries,
                "namespace": namespace,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.base_url}/query/batch",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def rebuild_vectorstore(self, namespace: str = None) -> Dict[str, Any]:
        """Reconstruye el índice vectorial"""
        try:
            params = {}
            if namespace:
                params['namespace'] = namespace
            
            response = self.session.post(
                f"{self.base_url}/vectorstore/rebuild",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def sync_documents(self) -> Dict[str, Any]:
        """Sincroniza documentos desde el directorio /pdfs"""
        try:
            response = self.session.post(f"{self.base_url}/documents/sync")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def print_section(title: str):
    """Imprime una sección con formato"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(operation: str, result: Dict[str, Any]):
    """Imprime el resultado de una operación"""
    print(f"\n🔹 {operation}")
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Éxito:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

def main():
    """Función principal de pruebas"""
    print("🚀 Iniciando pruebas del microservicio RAG")
    
    # Crear cliente
    client = RAGMicroserviceClient()
    
    # === PRUEBAS BÁSICAS ===
    print_section("PRUEBAS BÁSICAS")
    
    # Health check
    result = client.health_check()
    print_result("Health Check", result)
    
    # Estado del servicio
    result = client.get_status()
    print_result("Estado del Servicio", result)
    
    # Listar namespaces
    result = client.list_namespaces()
    print_result("Listar Namespaces", result)
    
    # === PRUEBAS DE CARGA DE DOCUMENTOS ===
    print_section("PRUEBAS DE CARGA DE DOCUMENTOS")
    
    # Buscar PDFs en el directorio actual o pdfs/
    pdf_paths = []
    for pdf_dir in ['.', './pdfs', '../pdfs']:
        pdf_dir = Path(pdf_dir)
        if pdf_dir.exists():
            pdf_paths.extend(list(pdf_dir.glob('*.pdf')))
    
    if pdf_paths:
        pdf_path = pdf_paths[0]
        print(f"📄 Usando PDF de prueba: {pdf_path}")
        
        result = client.upload_document(
            str(pdf_path), 
            namespace="test_namespace",
            description="Documento de prueba"
        )
        print_result(f"Subir Documento ({pdf_path.name})", result)
        
        # Esperar un poco para que se procese
        print("⏳ Esperando procesamiento...")
        time.sleep(5)
        
    else:
        print("⚠️  No se encontraron PDFs para prueba")
    
    # === PRUEBAS DE CONSULTAS ===
    print_section("PRUEBAS DE CONSULTAS")
    
    # Consulta simple
    test_queries = [
        "¿Cuáles son los puntos principales del documento?",
        "Resume el contenido",
        "¿Qué información relevante contiene?"
    ]
    
    for query in test_queries:
        result = client.query(
            query=query,
            namespace="test_namespace",
            max_results=3,
            temperature=0.7
        )
        print_result(f"Consulta: '{query[:50]}...'", result)
    
    # Consulta en lote
    result = client.batch_query(
        queries=test_queries[:2],
        namespace="test_namespace",
        max_results=2
    )
    print_result("Consulta en Lote", result)
    
    # === PRUEBAS DE CONFIGURACIÓN ===
    print_section("PRUEBAS DE CONFIGURACIÓN")
    
    # Sincronizar documentos
    result = client.sync_documents()
    print_result("Sincronizar Documentos", result)
    
    # Reconstruir vectorstore
    result = client.rebuild_vectorstore(namespace="test_namespace")
    print_result("Reconstruir Vectorstore", result)
    
    # === PRUEBAS DE DIFERENTES PARÁMETROS ===
    print_section("PRUEBAS CON DIFERENTES PARÁMETROS")
    
    if pdf_paths:  # Solo si tenemos documentos
        # Temperatura baja (más determinístico)
        result = client.query(
            query="Resume brevemente",
            namespace="test_namespace",
            temperature=0.1,
            max_tokens=200
        )
        print_result("Consulta con Temperatura Baja", result)
        
        # Temperatura alta (más creativo)
        result = client.query(
            query="Resume brevemente",
            namespace="test_namespace",
            temperature=1.5,
            max_tokens=200
        )
        print_result("Consulta con Temperatura Alta", result)
    
    # === RESUMEN FINAL ===
    print_section("RESUMEN FINAL")
    
    # Estado final
    result = client.get_status()
    if "error" not in result:
        print("✅ Microservicio funcionando correctamente")
        print(f"📊 Namespaces: {len(result.get('namespaces', []))}")
        print(f"🤖 Modelo de embeddings: {result.get('models', {}).get('embedding', 'N/A')}")
        print(f"🧠 Modelo de generación: {result.get('models', {}).get('generation', 'N/A')}")
    else:
        print("❌ Problemas detectados en el microservicio")
    
    print("\n🏁 Pruebas completadas")

def test_json_api():
    """Prueba específica de la API JSON"""
    print_section("PRUEBA DE API JSON")
    
    # Ejemplo de request JSON para consulta
    json_request = {
        "query": "¿Cuáles son las características principales?",
        "namespace": "compromidos",
        "max_results": 5,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    print("📨 Request JSON de ejemplo:")
    print(json.dumps(json_request, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json=json_request,
            timeout=TIMEOUT
        )
        
        print(f"\n📡 Status Code: {response.status_code}")
        print("📨 Response JSON:")
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pruebas del microservicio RAG")
    parser.add_argument("--url", default=BASE_URL, help="URL del microservicio")
    parser.add_argument("--json-only", action="store_true", help="Solo prueba JSON API")
    
    args = parser.parse_args()
    
    # Actualizar URL base
    BASE_URL = args.url
    
    if args.json_only:
        test_json_api()
    else:
        main()