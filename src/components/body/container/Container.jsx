import React from "react";
import ContainerCSS from "./Container.module.css";
import FileSelect from "./fileselect/FileSelect";
import PasswordModal from "../../modal/PasswordModal/PasswordModal";
import StepModal from "../../modal/StepModal/StepModal";

const Container = (props) => {
	return (
	<div>
		
		<div className={ContainerCSS.container}>
			<h1>Encrypt and Decrypt your Files using AES-256</h1>
			<h3>
				Protect your data by using AES 256 encryption by us. 
				Using AES-Encryption and our .cryption File Format, your Data is
				encrypted and decrypted securely in your browser.
			</h3>
			</div>
			<PasswordModal />
			<StepModal />

			<FileSelect />
		
		</div>
	
	);
};

export default Container;
