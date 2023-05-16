const button = document.querySelector('.button');
const bubbleContainer = document.createElement('div');
bubbleContainer.classList.add('bubble-container');
button.appendChild(bubbleContainer);

const infoList = ['Info 1', 'Info 2', 'Info 3']; // Liste d'informations

infoList.forEach((info, index) => {
  const bubble = document.createElement('div');
  bubble.classList.add('bubble');
  bubble.textContent = info;
  bubble.style.animationDelay = `${index * 0.2}s`; // Décalage d'animation pour chaque bulle
  bubbleContainer.appendChild(bubble);
});
