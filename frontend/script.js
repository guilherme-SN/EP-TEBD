document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("search-form");
  const imageInput = document.getElementById("image-input");
  const nResultsInput = document.getElementById("n-results");
  const gallery = document.getElementById("results-gallery");
  const loadingIndicator = document.getElementById("loading");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files[0];
    const nResults = nResultsInput.value;

    if (!file) {
      alert("Por favor, selecione uma imagem.");
      return;
    }

    // Mostra o indicador de "carregando" e limpa a galeria antiga
    loadingIndicator.classList.remove("hidden");
    gallery.innerHTML = "";

    const formData = new FormData();
    formData.append("image_file", file);
    formData.append("top_k", nResults);

    try {
      const response = await fetch(
        "http://0.0.0.0:8080/api/v1/embeddings/search",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`Erro na API: ${response.statusText}`);
      }

      const results = await response.json();

      // Esconde o indicador de "carregando"
      loadingIndicator.classList.add("hidden");

      if (results.length === 0) {
        gallery.innerHTML = "<p>Nenhuma imagem similar encontrada.</p>";
      } else {
        results.similar_embeddings.forEach((result, index) => {
          if (result && result.source) {
            console.log(`Processando item #${index + 1}:`, result);

            const itemDiv = document.createElement("div");
            itemDiv.className = "gallery-item";

            const img = document.createElement("img");
            img.src = result.source;
            img.alt = result.label;

            const p = document.createElement("p");
            p.textContent = result.label;

            itemDiv.appendChild(img);
            itemDiv.appendChild(p);
            gallery.appendChild(itemDiv);
          } else {
            console.error(
              `Item inválido encontrado no índice #${index}:`,
              result
            );
          }
        });
      }
    } catch (error) {
      loadingIndicator.classList.add("hidden");
      gallery.innerHTML = `<p style="color: red;">Ocorreu um erro: ${error.message}</p>`;
      console.error("Erro ao buscar imagens:", error);
    }
  });
});
