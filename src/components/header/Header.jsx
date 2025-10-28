import React from "react";
import { Layout, Icon } from "antd";
import headerCSS from "./header.module.css";

const Header = (props) => {
	return (
		<Layout.Header className={headerCSS.header}>
			<div className={headerCSS.brand}>
				<h1>Encryption and Decryption using AES 256 bits</h1>
			</div>
		</Layout.Header>
	);
};

export default Header;
