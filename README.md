# Smart Sync Audio Book

The project is designed for automatic generation of audiobooks in a foreign languages, with personalized translation and interval fixation of words. When generating an audiobook, each sentence is accompanied by translations of words and expressions that are new or have not met for a long time, according to the interval method, for a certain user, as well as a full translation of the sentence, if the sentence contains a lot of new words. These words and expressions are stored in a personal dictionary, and used in subsequent generations. The dictionary takes into account both specific forms of words and their basic lemmas, with different fixation intervals.

The input is the text in the original language. You can also add your own translation file to be used for generation, or leave this task to an external translator (currently the choice is between local Argos Translate and the Google Cloud Translation API). After matching words and expressions in the original and translation, speech synthesis is performed using the provider specified by the user with the specified settings of speaker, model, speed, …. Parts of the audio track with individual words for translation are generated separately, or can be additionally recognized from the generated full sentence using MFA (Montreal Forced Aligner) and reused in the dictionary, at the user's choice. The speed at which sentences and the dictionary are voiced is set separately by the user.