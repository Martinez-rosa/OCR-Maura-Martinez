**INFORME TÉCNICO**  
**Sistema OCR con Segmentación Clásica y CNN**

Trabajo de Visión por Computador / Reconocimiento de Patrones

Alumno: Maura Martínez Noda  
Asignatura: Inteligencia Artificial

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## **1\. Introducción**

El objetivo del proyecto es desarrollar un sistema OCR (Optical Character Recognition) capaz de reconocer texto digital y manuscrito en documentos escaneados o fotografiados, extraer texto línea por línea, detectar imágenes y proporcionar una interfaz gráfica de uso sencillo.

El sistema está diseñado para funcionar completamente en CPU y de forma local, sin depender de servicios externos, lo que lo hace adecuado para entornos educativos y administrativos.

Para abordar el problema opté por una arquitectura modular, separando claramente las fases de segmentación, reconocimiento, extracción de imágenes y presentación de resultados.

Decidí combinar técnicas clásicas de visión por computador para la segmentación con modelos de redes neuronales convolucionales (CNN) entrenados previamente para el reconocimiento de caracteres.

Además, se implementó una interfaz gráfica basada en Flask que permite al usuario cargar imágenes desde el navegador y visualizar tanto el texto reconocido como las imágenes extraídas del documento.

Los objetivos clave del sistema se definen de la siguiente manera:

* **Reconocimiento Dual:** Procesar con eficacia tanto texto digital como manuscrito presente en documentos escaneados o fotografiados.  
* **Extracción Estructurada:** Extraer el contenido textual organizándolo línea por línea para preservar el formato básico del documento original.  
* **Detección de Contenido Visual:** Identificar y extraer imágenes.  
* **Interfaz de Usuario Accesible:** Proveer una interfaz gráfica de usuario (GUI) sencilla e intuitiva que facilite la carga de documentos y la visualización de los resultados.

Para alcanzar estos objetivos, he adoptado una doble estrategia técnica que combina la fiabilidad de las técnicas clásicas de visión por computador con el poder de los modelos de aprendizaje profundo. Específicamente, el sistema utiliza un pipeline robusto de algoritmos de visión por computador, como la binarización adaptativa y el análisis de contornos, para la fase de segmentación de documentos (división en líneas y caracteres) y emplea una Red Neuronal Convolucional (CNN) para el reconocimiento de los caracteres individuales. Esta elección de diseño híbrido resulta en un enfoque modular, eficiente y fácilmente mantenible.

## **2\. Arquitectura y Diseño de la Solución**

La base del sistema es una arquitectura modular, una decisión de diseño estratégica que aporta flexibilidad y robustez. Este enfoque permite la especialización de cada componente en una tarea concreta, simplifica significativamente el mantenimiento y sienta las bases para futuras expansiones o sustituciones de módulos sin necesidad de rediseñar la solución completa.

El sistema se organiza en torno a un orquestador central que coordina los siguientes módulos principales:

* **`DocumentSegmenter`**: Responsable de analizar la estructura del documento para aislar las líneas de texto y los caracteres individuales.  
* **`CNNRecognizer`**: Encargado de identificar cada carácter segmentado utilizando un modelo de red neuronal convolucional.  
* **`ImageExtractor`**: Especializado en detectar y extraer contenido no textual, como fotografías o diagramas.  
* **Interfaz `Flask`**: Proporciona un servidor web ligero que ofrece una interfaz gráfica al usuario para interactuar con el sistema a través de un navegador.

### **2.1. Flujo de Procesamiento de Datos**

El recorrido de un documento a través del sistema sigue una secuencia de pasos bien definida, garantizando un procesamiento coherente y predecible desde la entrada hasta la salida.

1. **Carga del Documento:** El proceso se inicia cuando el usuario proporciona una imagen, ya sea cargándola a través de la interfaz web o especificando su ruta mediante la línea de comandos.  
2. **Orquestación y Normalización:** La imagen es recibida por el orquestador central (`OCRSystem`), que la normaliza y prepara para el procesamiento. En esta etapa, se determina el tipo de texto (digital o manuscrito) y si se debe activar la extracción de imágenes.  
3. **Segmentación de Líneas:** El módulo de segmentación analiza la imagen para identificar y separar las distintas líneas de texto. Para ello, agrupa los elementos detectados según su posición vertical.  
4. **Segmentación de Caracteres:** Una vez aislada cada línea, el sistema procede a dividirla en sus componentes individuales: los caracteres. Cada carácter se recorta y se normaliza a un tamaño estándar para la siguiente fase.  
5. **Reconocimiento de Caracteres:** Los caracteres normalizados se envían al módulo de reconocimiento, donde la red neuronal convolucional (CNN) predice la identidad de cada uno, generando el texto digital correspondiente.  
6. **Extracción de Imágenes:** De forma paralela, si la opción está activada, el módulo de extracción de imágenes analiza el documento en busca de regiones no textuales de alta complejidad visual, las aísla y las guarda como archivos independientes.  
7. **Generación de Resultados:** Finalmente, el orquestador unifica el texto reconocido y las imágenes extraídas. Los resultados se guardan en una carpeta local denominada `Output` y, si se utiliza la interfaz gráfica, se envían de vuelta al navegador para su visualización.

### **2.2. Descripción Detallada de Módulos del Núcleo (`src/core`)**

Los siguientes módulos constituyen el núcleo de procesamiento del sistema y encapsulan la lógica fundamental de la solución.

**`ocr_system.py` — El Orquestador Central**

Este módulo actúa como el controlador principal que dirige todo el flujo de trabajo. Su función no es realizar el procesamiento de bajo nivel, sino delegar las tareas a los módulos especializados y unificar los resultados. Al centralizar la coordinación, este diseño promueve una clara separación de responsabilidades, haciendo el sistema más legible y fácil de mantener. Sus entradas principales son `image_input` (la imagen a procesar), `text_type` (para seleccionar el modelo de reconocimiento adecuado) y `extract_images` (un indicador para activar o desactivar la extracción de contenido visual). Este patrón de diseño de orquestador central desacopla la lógica de negocio del procesamiento de bajo nivel, maximizando la modularidad y facilitando futuras sustituciones de componentes.

**`segmentation.py` — El Módulo de Segmentación**

La precisión del sistema OCR depende críticamente del rendimiento de este módulo. Su responsabilidad es "diseccionar" la imagen del documento en partes más pequeñas y manejables: primero en líneas de texto y luego en caracteres individuales. Para lograrlo, emplea una secuencia de técnicas clásicas de visión por computador, incluyendo la mejora de contraste con CLAHE, binarización adaptativa para aislar el texto del fondo, detección de contornos para identificar formas, y algoritmos de clustering como DBSCAN para agrupar los caracteres en líneas coherentes. La elección de un pipeline clásico para esta tarea crítica ofrece previsibilidad y eficiencia computacional en CPU, evitando la sobrecarga de un modelo neuronal para una tarea estructuralmente definida.

**`recognition.py` — El Módulo de Reconocimiento**

Este componente es el núcleo de inteligencia artificial del sistema. Su tarea es recibir una imagen normalizada de un carácter y determinar a qué letra o símbolo corresponde. El proceso principal se basa en una CNN que, tras recibir la imagen de un carácter, calcula un vector de probabilidades para todas las clases posibles (letras, números, símbolos) y selecciona aquella con la puntuación más alta. El módulo también contempla una estrategia alternativa de `Template Matching` (comparación de plantillas), aunque el enfoque neuronal es el principal. Esta especialización permite que el poder del deep learning se concentre exclusivamente en la clasificación, que es la tarea con mayor variabilidad y complejidad.

**`image_extraction.py` — El Módulo de Extracción de Contenido Visual**

Este módulo se especializa en identificar y aislar contenido no textual, como fotografías, logos o diagramas. Utiliza técnicas como los gradientes de Sobel y operaciones morfológicas para detectar regiones con alta varianza y complejidad visual, características típicas de las imágenes. Posteriormente, aplica filtros basados en el área, la proporción y la desviación estándar para discriminar estas regiones del texto circundante. Al aislar el contenido visual en un flujo paralelo, se previene la contaminación del pipeline de OCR y se genera un valor añadido al preservar los activos gráficos del documento.

**`preprocessor.py` — El Módulo de Preprocesamiento**

La calidad y la consistencia de los datos de entrada son fundamentales para el buen funcionamiento de cualquier sistema de reconocimiento de patrones. Este módulo se encarga de preparar las imágenes antes de que sean procesadas por los módulos de segmentación y reconocimiento. Sus responsabilidades clave incluyen la conversión a blanco y negro, la eliminación de ruido visual, la normalización de los caracteres a un tamaño y formato estándar, y el centrado geométrico de los caracteres. Esta estandarización rigurosa en la entrada es una inversión fundamental que mitiga la variabilidad y maximiza la precisión del pipeline de reconocimiento aguas abajo.

### **2.3. Módulos de Interfaz y Ejecución**

Estos componentes gestionan la interacción con el usuario y el arranque del sistema.

**`gui.py` — Servidor de Interfaz Gráfica**

Este módulo funciona como un backend web basado en el microframework Flask. Su responsabilidad es iniciar un servidor local que aloja la interfaz de usuario. Gestiona las peticiones HTTP del navegador, recibe las imágenes que el usuario carga, las envía al `OCRSystem` para su procesamiento y devuelve el texto reconocido y las imágenes extraídas para ser mostradas en el frontend. El módulo contiene también los activos HTML, CSS y JavaScript embebidos, lo que lo convierte en un componente autocontenido para la interfaz gráfica.

**`main.py` — Punto de Entrada del Sistema**

Este script es el punto de entrada principal para la ejecución del programa. Su rol es interpretar los argumentos proporcionados en la consola para determinar el modo de operación: interfaz gráfica (GUI) o línea de comandos (CLI). Una vez decidido el modo, prepara el entorno necesario (como la creación de carpetas de salida) y pone en marcha la ejecución del `OCRSystem`.

La descripción detallada de la arquitectura sienta las bases para comprender el rendimiento del sistema en la práctica, aspecto que se analiza a continuación.

## **3\. Evaluación de Rendimiento**

Medir el rendimiento de los distintos componentes del sistema es fundamental para identificar posibles cuellos de botella y optimizar la experiencia del usuario. Las pruebas se centraron en medir el tiempo de ejecución de las fases clave del proceso en un entorno de CPU estándar.

Los resultados promedio se resumen en la siguiente tabla:

| Componente del Proceso | Tiempo de Ejecución (segundos) |
| :---- | :---- |
| Carga de modelos | 3–6 |
| Segmentación | 0.4–0.8 |
| Reconocimiento | 0.8–1.5 |
| Extracción de imágenes | 0.2–0.8 |

El análisis de estos datos revela que la carga inicial de los modelos de reconocimiento en memoria constituye el principal coste de latencia del sistema. Sin embargo, este impacto es un coste único que se produce al iniciar la aplicación. En un escenario de uso continuo a través de la interfaz de usuario, donde los modelos permanecen cargados, la latencia por documento se reduce considerablemente, limitándose a la suma de los tiempos de segmentación, reconocimiento y extracción. Este perfil de rendimiento valida la idoneidad de la arquitectura para casos de uso interactivos y de procesamiento de un solo documento. Para operaciones de procesamiento por lotes a gran escala, la optimización de la carga de modelos o una arquitectura basada en servicios sería el siguiente paso lógico.

Este rendimiento se observó de manera consistente durante el ciclo de desarrollo, como se documenta en el siguiente registro de pruebas.

## **4\. Registro de Pruebas y Desarrollo Iterativo**

El siguiente registro es una evidencia del proceso de desarrollo iterativo seguido para construir y refinar el sistema. Cada entrada demuestra cómo las pruebas con diferentes tipos de documentos condujeron a ajustes específicos para mejorar la robustez y la funcionalidad general.

* `2026-01-11 16:22`: Se realizó una prueba con un documento complejo que incluía una fotografía. El sistema extrajo correctamente la imagen, pero se requirieron ajustes en los umbrales de detección del módulo de extracción para optimizar la precisión del recorte.  
* `2026-01-11 16:37`: Se llevaron a cabo tareas de refactorización de código para mejorar la legibilidad y se eliminaron módulos heredados obsoletos. Estas acciones contribuyeron a estabilizar el comportamiento de la interfaz de usuario.  
* `2026-01-11 16:39`: Se verificó el correcto funcionamiento del sistema a través de un arranque desde la GUI. El flujo de carga de imagen y visualización de resultados operó según lo esperado.  
* `2026-01-11 17:10`: Se ejecutó una prueba con un documento de texto manuscrito. El sistema logró un reconocimiento parcial, lo que puso de manifiesto las limitaciones del dataset utilizado para entrenar el modelo de CNN, que era reducido para este tipo de escritura.

Este ciclo de pruebas y ajustes fue clave para alcanzar la versión final del sistema, cuyas conclusiones y lecciones aprendidas se presentan a continuación.

## **5\. Conclusiones y Futuras Líneas de Trabajo**

El sistema OCR desarrollado cumple satisfactoriamente con los objetivos propuestos en su concepción. La integración de técnicas de segmentación clásica con un reconocedor basado en una Red Neuronal Convolucional ha demostrado ser un enfoque eficaz. La arquitectura modular no solo ha facilitado el desarrollo, sino que también garantiza la mantenibilidad y la escalabilidad del sistema a largo plazo, permitiendo la sustitución o mejora de componentes individuales sin afectar al resto de la estructura.

### **5.1. Limitaciones Identificadas**

Durante el desarrollo y las pruebas, se identificaron varias limitaciones inherentes al diseño actual del sistema:

* **Dependencia de la Calidad de la Segmentación:** La precisión global del OCR está fuertemente condicionada por el éxito de la etapa de segmentación. Errores al separar líneas o caracteres se propagan inevitablemente al módulo de reconocimiento.  
* **Dataset de Entrenamiento Reducido:** El rendimiento del reconocimiento de texto manuscrito es limitado debido al tamaño y la variedad del dataset utilizado para entrenar la CNN.  
* **Ausencia de Postprocesado Lingüístico:** El sistema no incluye un módulo de corrección ortográfica o de análisis contextual, por lo que los errores de reconocimiento a nivel de carácter no son corregidos.  
* **Incapacidad para Procesar Tablas:** El diseño actual no contempla la detección o extracción de datos estructurados en formato de tablas.

### **5.2. Mejoras Propuestas**

Para abordar las limitaciones identificadas y aumentar el valor de la solución, se proponen las siguientes líneas de trabajo futuras:

1. **Integración de Postprocesado Lingüístico:** Añadir un módulo que utilice modelos de lenguaje o diccionarios para corregir errores comunes y elevar la calidad del texto final de un reconocimiento de caracteres a un nivel de documento semánticamente coherente.  
2. **Paralelización del Procesamiento por Líneas:** Optimizar la velocidad de procesamiento en documentos largos mediante la paralelización del reconocimiento de caracteres para mejorar significativamente el throughput en escenarios de digitalización masiva.  
3. **Exportación a Formatos de Documento:** Implementar la capacidad de exportar los resultados directamente a formatos como PDF con texto seleccionable o Microsoft Word para integrar el sistema de forma nativa en flujos de trabajo administrativos y de oficina existentes.  
4. **Controles Avanzados en la Interfaz Gráfica:** Incorporar en la GUI opciones para que el usuario pueda ajustar parámetros clave, como los umbrales de binarización, para empoderar a los usuarios avanzados con la capacidad de optimizar el rendimiento en documentos particularmente desafiantes.

