# Pipeline de Optimización SEO para Felinos

Este repositorio automatiza la optimización SEO de contenidos publicados en WordPress usando datos maestros almacenados en Google Sheets, inteligencia SERP y generación asistida por IA. El flujo completo se ejecuta mediante un **GitHub Action** con una cadencia de **28 días** anclada al jueves 28-dic-2025 (ver detalle más abajo) o manualmente bajo demanda.

## Estructura principal

- `.github/workflows/optimizar_posts.yml`: workflow de GitHub Actions que instala dependencias, ejecuta el script y controla la ventana de ejecución (guardia de 28 días).
- `scripts/optimizar_posts.py`: orquestador Python 3.11 que conecta Google Sheets, WordPress, SERP API y OpenAI.
- `prompts/optimizar_post.txt`: plantilla que se envía al modelo de IA con todo el contexto necesario.
- `.env.example`: referencia de variables de entorno para ejecución local.
- `requirements.txt`: dependencias Python (incluye `httpx>=0.27` para compatibilidad con el SDK de OpenAI).

## Configuración de secretos y variables en GitHub

1. Ve a tu repositorio → **Settings → Secrets and variables → Actions**.
2. Crea los siguientes **secrets** obligatorios:
	- `GOOGLE_SERVICE_ACCOUNT_JSON`: JSON completo del servicio con permisos sobre el Google Sheet `SEOMasterDashboard_Felinos`.
	- `WP_URL`: URL base de tu sitio, ejemplo `https://tu-sitio.com`.
	- `WP_USER`: usuario con acceso a la API de WordPress.
	- `WP_APP_PASSWORD`: contraseña de aplicación generada en WordPress.
	- `SERP_API_KEY`: clave de SerpAPI (u otro proveedor compatible).
	- `OPENAI_API_KEY`: clave con acceso al modelo configurado.
3. (Opcional) En la pestaña **Variables**, define:
	- `MAX_POSTS_PER_RUN`: máximo de posts a procesar por ejecución (el workflow la fija actualmente en 10).
	- Si necesitas cambiar la zona horaria o el modelo, puedes añadir `PIPELINE_TIMEZONE` o `OPENAI_MODEL` aquí mismo.

## Requisitos de terceros

- **Google Sheets**: el servicio debe tener acceso de edición al documento `SEOMasterDashboard_Felinos` (ID `1Hwues5snSJFqJRTzEXFmts3N3OpO0Tgpejub_nosl40`).
- **WordPress**: habilita contraseñas de aplicación y asegúrate de que el usuario tenga permisos para editar entradas publicadas.
- **SerpAPI**: el workflow usa `engine=google` y `location=Colombia`. Si cambias de proveedor, ajusta `SERP_API_ENGINE` y `SERP_API_LOCATION` para que coincidan con la API elegida.
- **OpenAI**: confirma que tu cuenta soporte el modelo definido (`gpt-4.1` por defecto) y que no existan restricciones de cuota.

## Ejecución local (opcional)

1. Copia `.env.example` a `.env` y rellena los valores.
2. Crea un entorno virtual con Python 3.11 e instala dependencias:
	```bash
	python -m venv .venv
	source .venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt
	```
3. Ejecuta el pipeline en modo verbose:
	```bash
	source .env
	python scripts/optimizar_posts.py \
	  --prompt-file prompts/optimizar_post.txt \
	  --max-posts 3 \
	  --verbose
	```
4. Revisa la consola para confirmar la lectura de Google Sheets, la llamada a SERP y la actualización en WordPress.

## Flujo del pipeline

1. Lee `Sheet_Final` y filtra filas con `Ejecutar_Accion = "SI"` y al menos 30 días desde `Ultima_Optimización`.
2. Cruza cada fila con `indice_contenido` (por `Post_ID` o `URL`).
3. Consulta SERP (People Also Ask, competidores, búsquedas relacionadas, snippets).
4. Descarga el contenido actual del post en WordPress.
5. Construye el prompt con el contexto completo y lo envía a OpenAI.
6. Recibe HTML optimizado + metadatos SEO, lo publica directamente en WordPress.
7. Registra el cambio en `Logs_Optimización` y actualiza campos clave en el índice.
8. Genera un resumen JSON (`pipeline_summary.json`) con el total de posts optimizados/omitidos y notas del run; el paso final envía una notificación a Telegram usando ese resumen y el log (`pipeline.log`).

## Depuración

- Si el paso “Run SEO optimization pipeline” falla con errores de OpenAI relacionados con `proxies`, asegúrate de que `httpx>=0.27.0` esté instalado (ya está en `requirements.txt`).
- Activa el modo verbose mediante la opción `--verbose` (el workflow ya lo hace) para seguir la traza completa en GitHub Actions.
- El script detiene cada post que falle (SERP, WordPress, OpenAI) y continúa con el resto; revisa la lista de “Skipped items” al final del log.

## Ejecución programada y manual

- El workflow se despierta **todos los días a las 04:00 hora de Colombia (09:00 UTC)**, pero el paso `schedule-check` sólo permite continuar cuando han transcurrido múltiplos exactos de 28 días desde el **jueves 28-dic-2025**. Las próximas fechas válidas después de esa ancla son 25-ene-2026, 22-feb-2026, etc.
- Si el guardia detecta que todavía no toca ejecutar, el job principal se omite y se deja constancia en el log.
- Cualquier ejecución manual (botón **Run workflow**) omite el guardia de calendario y procesa los posts de inmediato.

Con los secretos configurados y las dependencias instaladas, el pipeline queda listo para mantener tu contenido optimizado sin intervención manual.
