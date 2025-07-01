import os
import asyncio
from tqdm.asyncio import tqdm

from schemas.requests.embedding_create import EmbeddingCreate
from services.embedding_service import EmbeddingService


async def process_images_in_directory(service: EmbeddingService, base_directory: str):
    """Percorre o diretório, extrai embeddings e salva no banco"""
    print(f"Iniciando varredura no diretório: {base_directory}")

    # Collect all image paths and labels first
    all_images = []
    all_labels = []
    
    for letter_folder in os.listdir(base_directory):
        letter_path = os.path.join(base_directory, letter_folder)
        
        if not os.path.isdir(letter_path):
            continue
            
        letter = letter_folder.upper()
        image_files = [f for f in os.listdir(letter_path) if f.lower().endswith('.png')]
        print(f"Encontrada letra: {letter} ({len(image_files)} imagens)")
        
        for image_filename in image_files:
            image_number = os.path.splitext(image_filename)[0]
            label = f"{letter}_{image_number}"
            image_full_path = os.path.join(letter_path, image_filename)
            
            all_images.append(image_full_path)
            all_labels.append(label)

    total_images = len(all_images)
    print(f"\nEncontradas {total_images} imagens. Processando em lotes...")

    # Process in batches
    batch_size = 16
    tasks = []
    
    for i in range(0, total_images, batch_size):
        batch_images = all_images[i:i + batch_size]
        batch_labels = all_labels[i:i + batch_size]
        
        task = asyncio.create_task(
            process_image_batch(service, batch_images, batch_labels)
        )
        tasks.append(task)

    for f in tqdm.as_completed(tasks):
        await f

async def process_image_batch(service: EmbeddingService, image_paths: list[str], labels: list[str]):
    # Extract embeddings in batch
    embeddings = await asyncio.to_thread(
        service.embedding_extractor.extract_embeddings_batch,
        image_paths
    )
    
    # Create tasks for saving to database
    save_tasks = []
    for embedding, label, image_path in zip(embeddings, labels, image_paths):
        embedding_create_data = EmbeddingCreate(
            embedding=embedding,
            label=label,
            source=image_path
        )
        save_tasks.append(service.create_embedding(embedding_create_data))
    
    # Wait for all saves to complete
    await asyncio.gather(*save_tasks)



async def main():
    train_directory = "../data/train"
    await process_images_in_directory(EmbeddingService(), train_directory)

    print("\nProcesso concluído com sucesso!")


asyncio.run(main())
