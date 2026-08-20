import React, { useState, useEffect } from 'react';
import './VirtualKeyboard.css';

const VirtualKeyboard = ({ language, onKeyPress, onClose }) => {
  const [activeKey, setActiveKey] = useState(null);
  // Keyboard layouts for each language
  const keyboards = {
    hindi: {
      name: 'Hindi (हिंदी)',
      rows: [
        // Numbers
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        // Vowels (स्वर)
        ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'अं', 'अः'],
        // Consonants Row 1 (व्यंजन)
        ['क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ'],
        // Consonants Row 2
        ['ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न'],
        // Consonants Row 3
        ['प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व'],
        // Consonants Row 4 & Special
        ['श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', 'श्र'],
        // Matras (मात्राएँ) & Symbols
        ['ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ं', 'ः', '।', '॥']
      ]
    },
    tamil: {
      name: 'Tamil (தமிழ்)',
      rows: [
        // Numbers
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        // Vowels (உயிரெழுத்து)
        ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ'],
        // Consonants Row 1 (மெய்யெழுத்து)
        ['க', 'ங', 'ச', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ப', 'ம'],
        // Consonants Row 2
        ['ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன'],
        // Additional Consonants
        ['ஜ', 'ஶ', 'ஷ', 'ஸ', 'ஹ', 'க்ஷ', 'ஶ்ரீ'],
        // Vowel Signs (உயிர்மெய் குறியீடுகள்)
        ['ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ'],
        // Special Characters & Numbers
        ['்', 'ஂ', 'ஃ', '।', '॥', '௦', '௧', '௨', '௩', '௪', '௫', '௬', '௭', '௮', '௯']
      ]
    },
    telugu: {
      name: 'Telugu (తెలుగు)',
      rows: [
        // Numbers
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        // Vowels (అచ్చులు)
        ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ౠ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ'],
        // Consonants Row 1 (హల్లులు)
        ['క', 'ఖ', 'గ', 'ఘ', 'ఙ', 'చ', 'ఛ', 'జ', 'ఝ', 'ఞ'],
        // Consonants Row 2
        ['ట', 'ఠ', 'డ', 'ఢ', 'ణ', 'త', 'థ', 'ద', 'ధ', 'న'],
        // Consonants Row 3
        ['ప', 'ఫ', 'బ', 'భ', 'మ', 'య', 'ర', 'ల', 'వ'],
        // Consonants Row 4 & Special
        ['శ', 'ష', 'స', 'హ', 'ళ', 'ఱ', 'క్ష', 'జ్ఞ'],
        // Vowel Signs (మాత్రలు)
        ['ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ'],
        // Special Characters & Numbers
        ['్', 'ం', 'ః', '।', '॥', '౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯']
      ]
    },
    kannada: {
      name: 'Kannada (ಕನ್ನಡ)',
      rows: [
        // Numbers
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        // Vowels (ಸ್ವರಗಳು)
        ['ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ'],
        // Consonants Row 1 (ವ್ಯಂಜನಗಳು)
        ['ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ'],
        // Consonants Row 2
        ['ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ'],
        // Consonants Row 3
        ['ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಲ', 'ವ'],
        // Consonants Row 4 & Special
        ['ಶ', 'ಷ', 'ಸ', 'ಹ', 'ಳ', 'ೞ', 'ಱ'],
        // Vowel Signs (ಮಾತ್ರೆಗಳು)
        ['ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ'],
        // Special Characters & Numbers
        ['್', 'ಂ', 'ಃ', '।', '॥', '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯']
      ]
    }
  };

  const currentKeyboard = keyboards[language];

  if (!currentKeyboard) {
    return null;
  }

  const handleKeyClick = (key) => {
    setActiveKey(key);
    onKeyPress(key);
    
    // Reset active key after animation
    setTimeout(() => setActiveKey(null), 150);
  };

  const handleSpace = () => {
    setActiveKey('SPACE');
    onKeyPress(' ');
    setTimeout(() => setActiveKey(null), 150);
  };

  const handleBackspace = () => {
    setActiveKey('BACKSPACE');
    onKeyPress('BACKSPACE');
    setTimeout(() => setActiveKey(null), 150);
  };

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="virtual-keyboard">
      <div className="keyboard-header">
        <h3>{currentKeyboard.name}</h3>
        <button
          className="close-keyboard"
          onClick={onClose}
          title="Close keyboard (Esc)"
        >
          ✕
        </button>
      </div>

      <div className="keyboard-body">
        {currentKeyboard.rows.map((row, rowIndex) => (
          <div key={rowIndex} className="keyboard-row">
            {row.map((key, keyIndex) => (
              <button
                key={keyIndex}
                className={`keyboard-key ${activeKey === key ? 'active' : ''}`}
                onClick={() => handleKeyClick(key)}
              >
                {key}
              </button>
            ))}
          </div>
        ))}

        <div className="keyboard-row keyboard-controls">
          <button
            className={`keyboard-key key-backspace ${activeKey === 'BACKSPACE' ? 'active' : ''}`}
            onClick={handleBackspace}
          >
            Backspace
          </button>
          <button
            className={`keyboard-key key-space ${activeKey === 'SPACE' ? 'active' : ''}`}
            onClick={handleSpace}
          >
            Space
          </button>
        </div>
      </div>
    </div>
  );
};

export default VirtualKeyboard;
